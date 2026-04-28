import cv2
import numpy as np
import os

def load_images(image_paths, target_width=800):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"❌ 오류: '{path}' 이미지를 찾을 수 없습니다.")
        else:
            h, w = img.shape[:2]
            target_height = int(h * (target_width / w))
            images.append(cv2.resize(img, (target_width, target_height)))
    return images

def blend_images(warped_base, warped_new):
    # 1. 흑백 마스크 생성 (검은색 여백은 0, 실제 이미지는 255)
    mask_base = cv2.cvtColor(warped_base, cv2.COLOR_BGR2GRAY)
    mask_base = (mask_base > 0).astype(np.uint8) * 255
    
    mask_new = cv2.cvtColor(warped_new, cv2.COLOR_BGR2GRAY)
    mask_new = (mask_new > 0).astype(np.uint8) * 255
    
    # 2. 거리 변환(Distance Transform) 적용
    # 이미지의 엣지(테두리)에서 중심부로 갈수록 값이 커지는 맵을 만듬
    dist_base = cv2.distanceTransform(mask_base, cv2.DIST_L2, 3)
    dist_new = cv2.distanceTransform(mask_new, cv2.DIST_L2, 3)
    
    # 3. 알파(가중치) 맵 계산
    # 0으로 나누는 것을 방지하기 위해 아주 작은 값(1e-5)을 더함
    sum_dist = dist_base + dist_new + 1e-5
    alpha_base = dist_base / sum_dist
    alpha_new = dist_new / sum_dist
    
    # RGB 3채널과 곱하기 위해 차원을 늘림
    alpha_base = np.expand_dims(alpha_base, axis=2)
    alpha_new = np.expand_dims(alpha_new, axis=2)
    
    # 4. 블렌딩 적용 (A * 가중치 + B * 가중치)
    blended_img = (warped_base * alpha_base + warped_new * alpha_new).astype(np.uint8)
    
    return blended_img

def stitch_images(images):
    if not images:
        return None

    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    base_img = images.pop(0)

    while images:
        print(f"\n🔄 남은 퍼즐 조각 {len(images)}장 중, 캔버스와 가장 잘 맞는 조각을 찾는 중...")

        gray_base = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
        _, mask_base = cv2.threshold(gray_base, 1, 255, cv2.THRESH_BINARY)
        kp_base, des_base = sift.detectAndCompute(base_img, mask_base)

        best_match_count = 0
        best_img_idx = -1
        best_good_matches = []
        best_kp_new = None

        for i, img in enumerate(images):
            gray_new = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask_new = cv2.threshold(gray_new, 1, 255, cv2.THRESH_BINARY)
            kp_new, des_new = sift.detectAndCompute(img, mask_new)

            if des_new is None or des_base is None:
                continue

            raw_matches = bf.knnMatch(des_new, des_base, k=2)

            good_matches = []
            for m, n in raw_matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > best_match_count:
                best_match_count = len(good_matches)
                best_img_idx = i
                best_good_matches = good_matches
                best_kp_new = kp_new

        if best_match_count < 15:
            print("⚠️ 더 이상 캔버스와 겹치는 사진을 찾을 수 없어 정합을 종료합니다.")
            break

        print(f"  -> 가장 잘 맞는 조각 발견! (매칭점: {best_match_count}개). 블렌딩 병합 시작...")

        best_img = images.pop(best_img_idx)

        pts_new = np.float32([best_kp_new[m.queryIdx].pt for m in best_good_matches]).reshape(-1, 2)
        pts_base = np.float32([kp_base[m.trainIdx].pt for m in best_good_matches]).reshape(-1, 2)

        H, status = cv2.findHomography(pts_new, pts_base, cv2.RANSAC, 5.0)

        if H is None:
            print("❌ 호모그래피 행렬 계산 실패. 이 조각은 건너뜁니다.")
            continue

        h_base, w_base = base_img.shape[:2]
        h_new, w_new = best_img.shape[:2]

        corners_new = np.float32([[0, 0], [0, h_new], [w_new, h_new], [w_new, 0]]).reshape(-1, 1, 2)
        warped_corners_new = cv2.perspectiveTransform(corners_new, H)
        corners_base = np.float32([[0, 0], [0, h_base], [w_base, h_base], [w_base, 0]]).reshape(-1, 1, 2)

        all_corners = np.concatenate((corners_base, warped_corners_new), axis=0)

        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

        result_width = x_max - x_min
        result_height = y_max - y_min

        if result_width > 15000 or result_height > 15000:
            print(f"❌ [경고] 비정상적인 팽창 발생 ({result_width} x {result_height}). 이 조각은 버립니다.")
            continue

        translation_dist = [-x_min, -y_min]
        H_translation = np.array([[1, 0, translation_dist[0]],
                                  [0, 1, translation_dist[1]],
                                  [0, 0, 1]])

        warped_new = cv2.warpPerspective(best_img, H_translation.dot(H), (result_width, result_height))
        
        warped_base = np.zeros_like(warped_new)
        warped_base[translation_dist[1]:h_base+translation_dist[1], translation_dist[0]:w_base+translation_dist[0]] = base_img

        base_img = blend_images(warped_base, warped_new)

    return base_img

def main():
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    image_list = [
        f for f in os.listdir('.') 
        if f.lower().endswith(valid_extensions) and 'stitched_result' not in f
    ]
    
    print(f"📂 감지된 이미지 파일 ({len(image_list)}장):")
    for img_name in image_list:
        print(f"   - {img_name}")

    if len(image_list) < 2:
        print("정합할 이미지가 최소 2장 이상 필요합니다. 폴더에 이미지를 추가해주세요.")
        return

    print("\n✅ 이미지를 불러오는 중...")
    images = load_images(image_list)

    final_result = stitch_images(images)

    if final_result is not None:
        cv2.imwrite('stitched_result.jpg', final_result)
        print("\n🎉 모든 정합 성공! 결과 이미지가 'stitched_result.jpg'로 저장되었습니다.")
        
        screen_img = cv2.resize(final_result, (800, int(final_result.shape[0] * (800 / final_result.shape[1]))))
        cv2.imshow('Blended Panorama', screen_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n❌ 정합에 실패했습니다.")

if __name__ == "__main__":
    main()