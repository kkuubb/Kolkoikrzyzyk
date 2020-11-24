import numpy as np
import cv2
from scipy import ndimage
from math import sqrt


class linia:
    def __init__(self, x1=0, y1=0, x2=0, y2=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.dlugosc = 0
        self.srodek = []
        self.pozycja = -1
        self.pozycjax = -1
        self.pozycjay = -1
        self.orientacja = -1

    def policzdlugosc(self):
        self.dlugosc = sqrt((self.x2-self.x1)**2+(self.y2-self.y1)**2)

    def policzsrodek(self):
        self.srodek = (self.x1+self.x2)/2, (self.y1+self.y2)/2
        #print(self.x1, self.x2, self.srodek[0])
    def sprawdzpozycje(self, szerokosc, wysokosc):
        self.pozycjax = self.srodek[0]/szerokosc
        self.pozycjay = self.srodek[1]/wysokosc
    def sprawdzorientacje(self):
        #print(abs(self.x1-self.x2), abs(self.y1-self.y2))
        if abs(self.x1-self.x2) > abs(self.y1-self.y2):
            self.orientacja = 0
        else:
            self.orientacja = 1





#create a 2d array to hold the gamestate
gamestate = [["-","-","-"],["-","-","-"],["-","-","-"]]

#kernel used for noise removal
kernel =  np.ones((7,7),np.uint8)
# Load a color image 
img = cv2.imread('7.png')
# get the image width and height
img_width = img.shape[1]
img_height = img.shape[0]



# turn into grayscale
img_g =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# turn into thresholded binary
ret,thresh1 = cv2.threshold(img_g,127,255,cv2.THRESH_BINARY)
#remove noise from binary
thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

kernel_size = 5
blur_gray = cv2.GaussianBlur(img_g,(kernel_size, kernel_size),0)

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

#kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(edges,kernel,iterations = 1)
erosion = cv2.erode(dilation,kernel,iterations = 1)

output = img.copy()
circles = cv2.HoughCircles(blur_gray,  cv2.HOUGH_GRADIENT, 1, 50, param1 = 50, param2 = 30, minRadius = 1, maxRadius = 100)
if circles is not None:
	circles = np.round(circles[0, :]).astype("int")
	for (x, y, r) in circles:
		cv2.circle(output, (x, y), r, (0, 255, 0), 4)
		cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    





#find and draw contours. RETR_EXTERNAL retrieves only the extreme outer contours
# contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# for i in range(len(contours)):
#     cv2.drawContours(img, contours, i, (0,255,0), 2)
#     cv2.imshow('image1', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 200  # minimum number of pixels making up a line
max_line_gap = 20 # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0 
gotowe = np.copy(img) * 0 # creating a blank to draw lines on
lines = cv2.HoughLinesP(erosion, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

linie = []
for i in lines:
    linie.append(linia(i[0][0], i[0][1], i[0][2], i[0][3]))

for i in linie:
    i.policzdlugosc()
    i.policzsrodek()
    i.sprawdzpozycje(img_width, img_height)
    i.sprawdzorientacje()

pion = []
poziom = []
pionprawo = []
pionlewo = []
poziomgora = []
poziomdol = []
for i in linie:
    if i.orientacja == 1 and i.pozycjax>=0.5:
        pionprawo.append(i)
        pion.append(i)
    if i.orientacja == 1 and i.pozycjax<0.5:
        pionlewo.append(i)
        pion.append(i)
    if i.orientacja == 0 and i.pozycjay<0.5:
        poziomgora.append(i)
        poziom.append(i)
    if i.orientacja == 0 and i.pozycjay>=0.5:
        poziomdol.append(i)
        poziom.append(i)

najdluzszapion = pion[0]
for i in pion:
    if i.dlugosc>najdluzszapion.dlugosc:
        najdluzszapion = i
najdluzszapoziom = poziom[0]
for i in poziom:
    if i.dlugosc>najdluzszapoziom.dlugosc:
        najdluzszapoziom = i
#print(najdluzszapion.dlugosc, najdluzszapoziom.dlugosc)


maksikx = pion[0]
maksiky = poziom[0]
minix = pion[0]
miniy = poziom[0]
for i in pionprawo:
    if i.pozycjax > maksikx.pozycjax and i.dlugosc>najdluzszapion.dlugosc-50:
        maksikx = i
for i in pionlewo:
    if i.pozycjax < minix.pozycjax and i.dlugosc>najdluzszapion.dlugosc-50:
        minix = i


for i in poziomdol:
    if i.pozycjay > maksiky.pozycjay and i.dlugosc>najdluzszapoziom.dlugosc-50:
        maksiky = i
for i in poziomgora:
    if i.pozycjay < miniy.pozycjay and i.dlugosc>najdluzszapoziom.dlugosc-50:
        miniy = i

#print(maksikx.pozycjax, maksiky.pozycjay, minix.pozycjax, miniy.pozycjay)

kreski = [maksikx, maksiky, minix, miniy]




for i in kreski:
    print(i.orientacja, i.pozycjax, i.pozycjay)
    cv2.line(gotowe, (i.x1,i.y1), (i.x2,i.y2),(255,255,0),5)
    # cv2.imshow('image1', gotowe)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()




for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,255,0),5)
# cv2.imshow('image1', line_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


lines_edges = cv2.addWeighted(img, 0.8, gotowe, 1, 0)
if circles is not None:
    for (x, y, r) in circles:
        cv2.circle(lines_edges, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(lines_edges, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)


cv2.imshow('image1', lines_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()