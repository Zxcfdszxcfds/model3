import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="图像特征检测与匹配平台", layout="wide")
st.title("📷 图像特征检测与匹配实验")

# ---------------------- 1. Canny边缘检测 ----------------------
st.header("1. Canny边缘检测（非极大值抑制对比）")
img_canny_file = st.file_uploader("上传图片（用于Canny边缘检测）", type=["jpg","png"], key="canny_up")

if img_canny_file:
    img_canny = cv2.imdecode(np.frombuffer(img_canny_file.read(), np.uint8), 1)
    img_canny = cv2.cvtColor(img_canny, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_canny, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    # 手动实现非极大值抑制前后对比
    edges_no_nms = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=False)
    edges_nms = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
    
    fig, axes = plt.subplots(1,3, figsize=(15,5))
    axes[0].imshow(img_canny)
    axes[0].set_title("原图")
    axes[0].axis("off")
    axes[1].imshow(edges_no_nms, cmap="gray")
    axes[1].set_title("无NMS边缘")
    axes[1].axis("off")
    axes[2].imshow(edges_nms, cmap="gray")
    axes[2].set_title("含NMS边缘")
    axes[2].axis("off")
    st.pyplot(fig)

# ---------------------- 2. Harris/SIFT特征点检测 ----------------------
st.header("2. Harris/SIFT特征点检测")
img_feat_file = st.file_uploader("上传图片（用于特征点检测）", type=["jpg","png"], key="feat_up")

if img_feat_file:
    img_feat = cv2.imdecode(np.frombuffer(img_feat_file.read(), np.uint8), 1)
    img_feat_rgb = cv2.cvtColor(img_feat, cv2.COLOR_BGR2RGB)
    gray_feat = cv2.cvtColor(img_feat, cv2.COLOR_BGR2GRAY)
    
    # Harris角点检测
    if st.button("检测Harris角点", key="harris_btn"):
        dst = cv2.cornerHarris(gray_feat, 2, 3, 0.04)
        dst = cv2.dilate(dst, None)
        img_harris = img_feat_rgb.copy()
        img_harris[dst > 0.01 * dst.max()] = [255,0,0]
        fig, ax = plt.subplots(figsize=(8,6))
        ax.imshow(img_harris)
        ax.set_title("Harris角点（红色标记）")
        ax.axis("off")
        st.pyplot(fig)
    
    # SIFT特征点检测
    if st.button("检测SIFT特征点", key="sift_btn"):
        sift = cv2.SIFT_create()
        kp = sift.detect(gray_feat, None)
        img_sift = cv2.drawKeypoints(img_feat_rgb, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.imshow(img_sift)
        ax.set_title("SIFT特征点（圆圈表示尺度）")
        ax.axis("off")
        st.pyplot(fig)

# ---------------------- 3. 图像匹配流程可视化 ----------------------
st.header("3. 图像匹配流程（特征点检测→匹配→RANSAC）")
img1_file = st.file_uploader("上传图像1", type=["jpg","png"], key="match_up1")
img2_file = st.file_uploader("上传图像2", type=["jpg","png"], key="match_up2")

if img1_file and img2_file:
    img1 = cv2.imdecode(np.frombuffer(img1_file.read(), np.uint8), 1)
    img2 = cv2.imdecode(np.frombuffer(img2_file.read(), np.uint8), 1)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    if st.button("执行图像匹配", key="match_btn"):
        # 特征点+描述子
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        # 初始匹配
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        
        # RANSAC计算单应矩阵
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # 可视化匹配结果
        matchesMask = mask.ravel().tolist()
        draw_params = dict(matchColor = (0,255,0),
                           singlePointColor = None,
                           matchesMask = matchesMask,
                           flags = 2)
        img_match = cv2.drawMatches(img1_rgb, kp1, img2_rgb, kp2, good, None, **draw_params)
        
        fig, ax = plt.subplots(figsize=(15,8))
        ax.imshow(img_match)
        ax.set_title("SIFT匹配 + RANSAC优化（绿色为有效匹配）")
        ax.axis("off")
        st.pyplot(fig)

# ---------------------- 4. 图像全景拼接 ----------------------
st.header("4. 多幅图像全景拼接")
img_pano_files = st.file_uploader("上传多张重叠图像", type=["jpg","png"], accept_multiple_files=True, key="pano_up")

if img_pano_files and len(img_pano_files)>=2:
    if st.button("生成全景图", key="pano_btn"):
        imgs = []
        for file in img_pano_files:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            imgs.append(img)
        
        stitcher = cv2.Stitcher_create()
        status, pano = stitcher.stitch(imgs)
        
        if status == cv2.Stitcher_OK:
            pano_rgb = cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(15,5))
            ax.imshow(pano_rgb)
            ax.set_title("全景拼接结果")
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.error("拼接失败：图像重叠不足或不匹配")

st.markdown("---")
st.caption("模式识别与图像处理 - A3作业平台")
