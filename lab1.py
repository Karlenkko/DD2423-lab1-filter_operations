import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from scipy.signal import convolve2d

from Functions import *
from gaussfft import gaussfft
from fftwave import fftwave


# Either write your code in a file like this or use a Jupyter notebook.
#
# A good idea is to use switches, so that you can turn things on and off
# depending on what you are working on. It should be fairly easy for a TA
# to go through all parts of your code though.
def fftwave(u, v, sz = 128):
	Fhat = np.zeros([sz, sz])
	Fhat[u, v] = 1
	F = np.fft.ifft2(Fhat)
	Fabsmax = np.max(np.abs(F))
	f = plt.figure(dpi=200)
	f.subplots_adjust(wspace=0.2, hspace=0.4)
	plt.rc('axes', titlesize=10)
	a1 = f.add_subplot(3, 2, 1)
	showgrey(Fhat, False)
	a1.title.set_text("Fhat: (u, v) = (%d, %d)" % (u, v))
	# What is done by these instructions?
	# make a translation move to make the origin in the center of the graph
	if u < sz/2:
		uc = u
	else:
		uc = u - sz
	if v < sz/2:
		vc = v
	else:
		vc = v - sz

	w1 = 2 * np.pi / (sz / uc)
	w2 = 2 * np.pi / (sz / vc)
	wavelength = 2 * np.pi / np.sqrt(w1 * w1 + w2 * w2) # Replace by correct expression
	amplitude = Fabsmax # Replace by correct expression
	a2 = f.add_subplot(3, 2, 2)
	showgrey(np.fft.fftshift(Fhat), False)
	a2.title.set_text("centered Fhat: (uc, vc) = (%d, %d)" % (uc, vc))
	a3 = f.add_subplot(3, 2, 3)
	showgrey(np.real(F), False, 64, -Fabsmax, Fabsmax)
	a3.title.set_text("real(F)")
	a4 = f.add_subplot(3, 2, 4)
	showgrey(np.imag(F), False, 64, -Fabsmax, Fabsmax)
	a4.title.set_text("imag(F)")
	a5 = f.add_subplot(3, 2, 5)
	showgrey(np.abs(F), False, 64, -Fabsmax, Fabsmax)
	a5.title.set_text("abs(F) (amplitude %f)" % amplitude)
	a6 = f.add_subplot(3, 2, 6)
	showgrey(np.angle(F), False, 64, -np.pi, np.pi)
	a6.title.set_text("angle(F) (wavelength %f)" % wavelength)
	plt.show()

# Exercise 1.3
def fourierTransformTest():
	fftwave(17, 9)

def linearity():
	F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
	G = F.T
	H = F + 2 * G

	f = plt.figure()
	f.subplots_adjust(wspace=0.2, hspace=0.2)
	plt.rc('axes', titlesize=10)
	a1 = f.add_subplot(3, 3, 1)
	showgrey(F, False)
	a1.title.set_text("F")
	a2 = f.add_subplot(3, 3, 2)
	showgrey(G, False)
	a2.title.set_text("G")
	a3 = f.add_subplot(3, 3, 3)
	showgrey(H, False)
	a3.title.set_text("H")

	Fhat = fft2(F)
	Ghat = fft2(G)
	Hhat = fft2(H)
	a4 = f.add_subplot(3, 3, 4)
	showgrey(np.log(1 + np.abs(Fhat)), False)
	a4.title.set_text("Fhat")
	a5 = f.add_subplot(3, 3, 5)
	showgrey(np.log(1 + np.abs(Ghat)), False)
	a5.title.set_text("Ghat")
	a6 = f.add_subplot(3, 3, 6)
	showgrey(np.log(1 + np.abs(Hhat)), False)
	a6.title.set_text("Hhat")
	a7 = f.add_subplot(3, 3, 7)
	showgrey(np.log(1 + np.abs(fftshift(Fhat))), False)
	a7.title.set_text("shifted Fhat")
	a8 = f.add_subplot(3, 3, 8)
	showgrey(np.log(1 + np.abs(fftshift(Ghat))), False)
	a8.title.set_text("shifted Ghat")
	a9 = f.add_subplot(3, 3, 9)
	showgrey(np.log(1 + np.abs(fftshift(Hhat))), False)
	a9.title.set_text("shifted Hhat")
	plt.show()

def multiplication():
	F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
	G = F.T

	f = plt.figure()
	f.subplots_adjust(wspace=0.2, hspace=0.2)
	plt.rc('axes', titlesize=10)
	a1 = f.add_subplot(1, 3, 1)
	showgrey(F * G, False)
	a1.title.set_text("F * G")
	a2 = f.add_subplot(1, 3, 2)
	showfs(fft2(F * G), False)
	a2.title.set_text("fft2(F * G)")

	Fhat = fftshift(fft2(F))
	Ghat = fftshift(fft2(G))
	a3 = f.add_subplot(1, 3, 3)
	showgrey(np.log(1 + np.abs(convolve2d(Fhat, Ghat, "same")/(F.shape[0]*F.shape[1]))), False)
	a3.title.set_text("conv2d(Fhat, Ghat)")
	plt.show()

def scaling():
	F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
		np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)
	# F = np.concatenate([np.zeros((56, 128)), np.ones((16, 128)), np.zeros((56, 128))])
	# Fbis = np.concatenate([np.zeros((48, 128)), np.ones((32, 128)), np.zeros((48, 128))])
	f = plt.figure()
	f.subplots_adjust(wspace=0.2, hspace=0.2)
	plt.rc('axes', titlesize=10)

	a1 = f.add_subplot(2, 2, 1)
	showgrey(F, False)
	a1.title.set_text("F")
	a2 = f.add_subplot(2, 2, 2)
	showfs(fft2(F), False)
	a2.title.set_text("fft2(F)")

	# a3 = f.add_subplot(2, 2, 3)
	# showgrey(Fbis, False)
	# a3.title.set_text("Fbis")
	# a4 = f.add_subplot(2, 2, 4)
	# showfs(fft2(Fbis), False)
	# a4.title.set_text("fft2(Fbis)")
	plt.show()

def rotation():
	alpha = [30, 45, 60, 90]
	F = np.concatenate([np.zeros((60, 128)), np.ones((8, 128)), np.zeros((60, 128))]) * \
		np.concatenate([np.zeros((128, 48)), np.ones((128, 32)), np.zeros((128, 48))], axis=1)
	f = plt.figure(figsize=(6, 8))
	for i in range(0, 4):
		G = rot(F, alpha[i])
		f.subplots_adjust(wspace=0.2, hspace=0.2)
		plt.rc('axes', titlesize=10)
		a1 = f.add_subplot(4, 3, 3 * i + 1)
		showgrey(G, False)
		a1.title.set_text("G" + str(i) + ", " + str(alpha[i]))
		a2 = f.add_subplot(4, 3, 3 * i + 2)
		Ghat = fft2(G)

		showfs(Ghat, False)
		a2.title.set_text("fft2(G)")
		Hhat = rot(fftshift(Ghat), -alpha[i])
		a3 = f.add_subplot(4, 3, 3 * i + 3)
		showgrey(np.log(1 + abs(Hhat)), False)
		a3.title.set_text("rotate Ghat")

	plt.show()

def phaseAndMagnitude():
	img1 = np.load("Images-npy/phonecalc128.npy")
	img2 = np.load("Images-npy/few128.npy")
	img3 = np.load("Images-npy/nallo128.npy")

	f = plt.figure()
	f.subplots_adjust(wspace=0.2, hspace=0.2)
	plt.rc('axes', titlesize=10)

	a1 = f.add_subplot(3, 3, 1)
	showgrey(img1, False)
	a1.title.set_text("phonecalc128")
	a2 = f.add_subplot(3, 3, 2)
	showgrey(img2, False)
	a2.title.set_text("few128")
	a3 = f.add_subplot(3, 3, 3)
	showgrey(img3, False)
	a3.title.set_text("nallo128")

	a4 = f.add_subplot(3, 3, 4)
	showgrey(pow2image(img1), False)
	a4.title.set_text("p2i phonecalc128")
	a5 = f.add_subplot(3, 3, 5)
	showgrey(pow2image(img2), False)
	a5.title.set_text("p2i few128")
	a6 = f.add_subplot(3, 3, 6)
	showgrey(pow2image(img3), False)
	a6.title.set_text("p2i nallo128")

	a7 = f.add_subplot(3, 3, 7)
	showgrey(randphaseimage(img1), False)
	a7.title.set_text("rpi phonecalc128")
	a8 = f.add_subplot(3, 3, 8)
	showgrey(randphaseimage(img2), False)
	a8.title.set_text("rpi few128")
	a9 = f.add_subplot(3, 3, 9)
	showgrey(randphaseimage(img3), False)
	a9.title.set_text("rpi nallo128")

	plt.show()

def gaussianfft(pic, t = 0.3):
	Fhat = fft2(pic)
	width, height = np.shape(pic)
	xRange = range(int(-width / 2), int(-width / 2) + width)
	yRange = range(int(-height / 2), int(-height / 2) + height)
	x, y = np.meshgrid(xRange, yRange)
	gaussianFilter = (1 / (2 * np.pi * t)) * np.exp(-(x * x + y * y) / (2 * t))
	gaussianFilter = gaussianFilter / sum(sum(gaussianFilter))
	filterhat = abs(fft2(gaussianFilter))

	res = Fhat * filterhat
	return abs(ifft2(res))

def gaussianTest():
	pic = deltafcn(128, 128)
	ts = [0.1, 0.3, 1, 10, 100]
	f = plt.figure()
	f.subplots_adjust(wspace=0.2, hspace=0.2)
	plt.rc('axes', titlesize=10)

	pos = 1
	for t in ts:
		# effect of discretization when t <= 0.3
		psf = gaussfft(deltafcn(128, 128), t)
		discpsf = discgaussfft(deltafcn(128, 128), t)
		a1 = f.add_subplot(1, len(ts), pos)
		showgrey(gaussianfft(psf, t), False)
		a1.title.set_text(t)
		pos += 1
		print("With t = ", t, ", gaussfft has a variance of ", variance(psf))
		print("With t = ", t, ", discgaussfft has a variance of ", variance(discpsf))

	plt.show()


def gaussianConvolution():
	gaussianTest()

	imgs = [np.load("Images-npy/phonecalc128.npy"),
		   np.load("Images-npy/few128.npy"),
		   np.load("Images-npy/nallo128.npy")]
	ts = [1, 4, 16, 64, 256]
	f = plt.figure(dpi=200)
	f.subplots_adjust(wspace=0.2, hspace=0.2)
	plt.rc('axes', titlesize=10)

	pos = 1
	for img in imgs:
		for t in ts:
			a1 = f.add_subplot(len(imgs), len(ts), pos)
			showgrey(gaussianfft(img, t), False)
			a1.title.set_text(t)
			pos += 1

	plt.show()

def smoothing():
	office = np.load("Images-npy/office256.npy")
	add = gaussnoise(office, 16)
	sap = sapnoise(office, 0.1, 255)
	picname = ["orig", "gsnoise", "sap"]
	imgs = [office, add, sap]
	smooths = [gaussianfft, discgaussfft, medfilt, ideal]
	gaussParam = [0.1, 0.2, 0.5, 0.8, 1, 2, 3, 5]
	medParam = range(1, 9)
	idealParam = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1]
	params = [gaussParam, gaussParam, medParam, idealParam]
	name = ["gauss", "discgauss", "med", "ideal"]
	f = plt.figure(figsize=(20, 18), dpi=320)
	f.subplots_adjust(wspace=0.2, hspace=0.2)
	plt.rc('axes', titlesize=10)
	pos = 1
	imgnum = 0
	for img in imgs:
		filtnum = 0
		for filt in smooths:
			for param in params[filtnum]:
				a1 = f.add_subplot(int(len(imgs) * len(smooths)), len(params[filtnum]), pos)
				showgrey(filt(img, param), False)
				title = picname[imgnum] + " " + name[filtnum] + ", " + str(param)
				a1.title.set_text(title)
				pos += 1

			filtnum += 1
		imgnum += 1
	plt.show()

def smoothAndSubsampling():
	img = np.load("Images-npy/phonecalc256.npy")
	smoothimg = img
	N = 5
	f = plt.figure()
	f.subplots_adjust(wspace=0, hspace=0)
	for i in range(N):
		if i > 0:  # generate subsampled versions
			img = rawsubsample(img)
			# smoothimg =  gaussfft(smoothimg, 0.5)# <call_your_filter_here>(smoothimg, <params>)
			smoothimg = ideal(smoothimg, 0.3)
			smoothimg = rawsubsample(smoothimg)
		a1 = f.add_subplot(3, N, i + 1)
		showgrey(img, False)
		if i == 0:
			a1.title.set_text("direct subsampling")
		a2 = f.add_subplot(3, N, i + N + 1)
		showgrey(smoothimg, False)
		if i == 0:
			a2.title.set_text("ideal, 0.3")

	img = np.load("Images-npy/phonecalc256.npy")
	smoothimg = img
	for i in range(N):
		if i > 0:  # generate subsampled versions
			img = rawsubsample(img)
			smoothimg =  gaussfft(smoothimg, 0.5)# <call_your_filter_here>(smoothimg, <params>)
			# smoothimg = ideal(smoothimg, 0.05)
			smoothimg = rawsubsample(smoothimg)
		a3 = f.add_subplot(3, N, i + 2*N + 1)
		showgrey(smoothimg, False)
		if i == 0:
			a3.title.set_text("gaussian, 0.5")
	plt.show()

if __name__ == '__main__':
	# fourierTransformTest()
	# linearity()
	# multiplication()
	# scaling()
	# rotation()
	phaseAndMagnitude()
	# gaussianConvolution()
	# gaussianTest()
	# smoothing()
	# smoothAndSubsampling()