import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

def detect_ridges(gray, sigma=3.0):
	H_elems = hessian_matrix(gray, sigma=sigma, order='rc')
	maxima_ridges, minima_ridges = hessian_matrix_eigvals(H_elems)
	return maxima_ridges, minima_ridges

def plot_images(*images):
	images = list(images)
	n = len(images)
	fig, ax = plt.subplots(ncols=n, sharey=True)
	for i, img in enumerate(images):
		ax[i].imshow(img, cmap='gray')
		ax[i].axis('off')
		extent = ax[i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
		plt.savefig('fig'+str(i)+'.png', bbox_inches=extent)
	plt.subplots_adjust(left=0.03, bottom=0.03, right=0.97, top=0.97)
	plt.show()

def main():
	# -------------------------- Step 1: import the image whose background has been removed ----------
	img = cv2.imread("input.jpg",1)

	# -------------------------- Step 2: Sharpen the image -------------------------------------------
	kernel = np.array([[-1,-1,-1], 
                   [-1, 9,-1],
                   [-1,-1,-1]])
	sharpened = cv2.filter2D(img, -1, kernel)
	# cv2.imshow("sharpened",sharpened)

	# --------------------------- Step 3: Grayscale the image------------------------------------------
	gray = cv2.cvtColor(sharpened,cv2.COLOR_BGR2GRAY)
	# cv2.imshow("gray",gray)

	# --------------------------- Step 4: Perform histogram equilisation ------------------------------
	hist,bins = np.histogram(gray.flatten(),256,[0,256])
	cdf = hist.cumsum()
	cdf_normalized = cdf * hist.max()/ cdf.max()

	cdf_m = np.ma.masked_equal(cdf,0)
	cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
	cdf = np.ma.filled(cdf_m,0).astype('uint8')

	img2=cdf[gray]
	# cv2.imshow("histogram",img2)
	# cv2.imwrite('hist.jpeg',img2)

	# ----------------------------- Step 5: Ridge detection filter ------------------------------------
	#sigma = 2.7
	a, b = detect_ridges(img2, sigma=2.7)
	plot_images(a, b)

	# ----------------------------- Step 6: Convert image to binary image -----------------------------
	img = cv2.imread('fig1.png',0)
	# cv2.imshow("img",img)
	bg = cv2.dilate(img,np.ones((5,5),dtype=np.uint8))
	bg = cv2.GaussianBlur(bg,(5,5),1)
	# cv2.imshow("bg",bg)
	src_no_bg = 255 - cv2.absdiff(img,bg)
	# cv2.imshow("src_no_bg",src_no_bg)
	ret,thresh = cv2.threshold(src_no_bg,240,255,cv2.THRESH_BINARY)
	cv2.imshow("threshold",thresh)

	# --------------------------- Step 7: Thinning / Skeletonizing Algorithm ----------------------------
	thinned = cv2.ximgproc.thinning(thresh)
	cv2.imshow("thinned",thinned)
	cv2.imwrite("./trial-out.png",thinned)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__=='__main__':
	main()