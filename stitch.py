import cv2
import numpy as np
import os

def stitch_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"오류: '{path}' 이미지를 찾을 수 없습니다.")
            return None
        
        # 해상도를 1000px로 설정 (디테일과 연산 속도의 타협점)
        h, w = img.shape[:2]
        new_w = 1000
        new_h = int(h * (new_w / w))
        img_resized = cv2.resize(img, (new_w, new_h))
        images.append(img_resized)

    print("✅ 모든 이미지 로드 완료! 정합을 시작합니다...\n")

    base_img = images[0]
    sift = cv2.SIFT_create()
    
    for i in range(1, len(images)):
        print(f"🔄 [ {i} / {len(images)-1} ] 번째 이미지와 이전 결과를 병합 중...")
        next_img = images[i]
        
        kp1, des1 = sift.detectAndCompute(base_img, None)
        kp2, des2 = sift.detectAndCompute(next_img, None)
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw_matches = bf.knnMatch(des2, des1, k=2)
        
        good_matches = []
        # 매칭 조건을 다시 조금 엄격하게(0.75) 변경하여 엉뚱한 점이 엮이는 것을 방지
        for m, n in raw_matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
                
        print(f"  -> 찾은 유효한 매칭점 개수: {len(good_matches)}개")
        
        if len(good_matches) >= 10:
            pts2 = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
            pts1 = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
            
            H, status = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
            
            if H is None:
                print(f"❌ 호모그래피 행렬을 계산할 수 없습니다 (이미지 {i}).")
                break

            h1, w1 = base_img.shape[:2]
            h2, w2 = next_img.shape[:2]
            
            corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
            warped_corners2 = cv2.perspectiveTransform(corners2, H)
            corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
            
            all_corners = np.concatenate((corners1, warped_corners2), axis=0)
            
            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            
            result_width = x_max - x_min
            result_height = y_max - y_min
            
            # 🚨 메모리 폭발 방지 안전장치 🚨
            # 계산된 캔버스 크기가 15000 픽셀을 넘어가면 엉뚱하게 매칭된 것으로 간주하고 중단
            if result_width > 15000 or result_height > 15000:
                print(f"❌ [경고] 비정상적인 매칭 발생! 생성될 이미지가 너무 큽니다 ({result_width} x {result_height}).")
                print("  -> 이미지 순서가 잘못되었거나, 특징점이 없는 하늘/바다 등이 너무 많이 겹쳤을 수 있습니다.")
                break

            translation_dist = [-x_min, -y_min]
            H_translation = np.array([[1, 0, translation_dist[0]], 
                                      [0, 1, translation_dist[1]], 
                                      [0, 0, 1]])
            
            try:
                output_img = cv2.warpPerspective(next_img, H_translation.dot(H), (result_width, result_height))
                output_img[translation_dist[1]:h1+translation_dist[1], translation_dist[0]:w1+translation_dist[0]] = base_img
                base_img = output_img
                print("  -> 병합 완료!\n")
            except Exception as e:
                print(f"❌ 이미지 와핑 중 에러 발생: {e}")
                break
        else:
            print(f"❌ 매칭점이 너무 적습니다 ({len(good_matches)}개).")
            break
            
    return base_img

image_list = ['stitchImage_1.jpg', 'stitchImage_2.jpg', 'stitchImage_3.jpg', 'stitchImage_4.jpg', 'stitchImage_5.jpg'] 
result = stitch_images(image_list)

if result is not None:
    cv2.imwrite('stitched_result.jpg', result)
    print("🎉 결과 이미지가 'stitched_result.jpg'로 성공적으로 저장되었습니다!")
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()