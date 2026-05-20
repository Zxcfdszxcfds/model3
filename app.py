import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="图像特征检测与匹配（无OpenCV版）", layout="wide")
st.title("📷 图像特征检测与匹配实验（纯Python实现）")

# ---------------------- 1. 简化版Canny边缘检测 ----------------------
st.header("1. 简化Canny边缘检测")
img_canny_file = st.file_uploader("上传图片", type=["jpg","png"], key="canny_up")

if img_canny_file:
    img = Image.open(img_canny_file).convert("RGB")
    img_np = np.array(img)
    gray = np.dot(img_np[...,:3], [0.299, 0.587, 0.114])
    
    # 简化边缘检测（模拟Canny效果）
    if st.button("生成边缘图", key="canny_btn"):
        # 水平和垂直梯度
        dx = np.abs(np.diff(gray, axis=1, prepend=gray[:,0:1]))
        dy = np.abs(np.diff(gray, axis=0, prepend=gray[0:1,:]))
        edge = dx + dy
        
        # 非极大值抑制简化模拟
        edge_nms = edge.copy()
        edge_nms[edge < 20] = 0
        
        fig, axes = plt.subplots(1,3, figsize=(15,5))
        axes[0].imshow(img_np)
        axes[0].set_title("原图")
        axes[0].axis("off")
        axes[1].imshow(edge, cmap="gray")
        axes[1].set_title("无NMS边缘")
        axes[1].axis("off")
        axes[2].imshow(edge_nms, cmap="gray")
        axes[2].set_title("含NMS边缘")
        axes[2].axis("off")
        st.pyplot(fig)

# ---------------------- 2. 简化Harris角点检测 ----------------------
st.header("2. 简化Harris角点检测")
img_harris_file = st.file_uploader("上传图片", type=["jpg","png"], key="harris_up")

if img_harris_file:
    img = Image.open(img_harris_file).convert("RGB")
    img_np = np.array(img)
    gray = np.dot(img_np[...,:3], [0.299, 0.587, 0.114])
    
    if st.button("检测角点", key="harris_btn"):
        # 简化角点检测（局部方差法）
        h, w = gray.shape
        corner = np.zeros_like(gray)
        # 局部窗口计算方差
        for i in range(1, h-1):
            for j in range(1, w-1):
                window = gray[i-1:i+2, j-1:j+2]
                var = np.var(window)
                if var > 500:
                    corner[i,j] = 255
        
        # 标记角点
        img_corner = img_np.copy()
        y, x = np.where(corner > 200)
        for i in range(len(x)):
            # 画圆圈标记角点
            plt.Circle((x[i], y[i]), 5, color="red", fill=False)
        
        fig, ax = plt.subplots(figsize=(8,6))
        ax.imshow(img_np)
        ax.scatter(x, y, s=20, c="red", marker="o")
        ax.set_title("简化Harris角点（红色标记）")
        ax.axis("off")
        st.pyplot(fig)

# ---------------------- 3. 简化图像匹配（直方图对比） ----------------------
st.header("3. 简化图像匹配演示")
img1_file = st.file_uploader("上传图像1", type=["jpg","png"], key="match_up1")
img2_file = st.file_uploader("上传图像2", type=["jpg","png"], key="match_up2")

if img1_file and img2_file:
    img1 = Image.open(img1_file).convert("RGB")
    img2 = Image.open(img2_file).convert("RGB")
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    
    if st.button("计算相似度", key="match_btn"):
        # 简化直方图匹配
        hist1, _ = np.histogram(img1_np.flatten(), bins=256, range=(0,255))
        hist2, _ = np.histogram(img2_np.flatten(), bins=256, range=(0,255))
        similarity = np.corrcoef(hist1, hist2)[0,1]
        
        fig, axes = plt.subplots(1,2, figsize=(12,5))
        axes[0].imshow(img1_np)
        axes[0].set_title("图像1")
        axes[0].axis("off")
        axes[1].imshow(img2_np)
        axes[1].set_title("图像2")
        axes[1].axis("off")
        st.pyplot(fig)
        
        st.success(f"图像相似度: {similarity:.2f}（越接近1越相似）")

st.markdown("---")
st.caption("模式识别与图像处理 - A3作业轻量版（无OpenCV依赖）")
