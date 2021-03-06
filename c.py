import numpy as np
import cv2
from scipy import ndimage
from math import sqrt

iksy = ['zdjecia/x1.png', 'zdjecia/x2.png', 'zdjecia/x3.png', 'zdjecia/x4.png']


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
        self.dlugosc = sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    def policzsrodek(self):
        self.srodek = (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def sprawdzpozycje(self, szerokosc, wysokosc):
        self.pozycjax = self.srodek[0] / szerokosc
        self.pozycjay = self.srodek[1] / wysokosc

    def sprawdzorientacje(self):
        if abs(self.x1 - self.x2) > abs(self.y1 - self.y2):
            self.orientacja = 0
        else:
            self.orientacja = 1


class cross:
    def __init__(self, x1=0, y1=0, x2=0, y2=0):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.srodek = []
        self.pozycja = '-1'

    def policzsrodek(self):
        self.srodek = (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def okreslpozycje(self, prawo, dol, lewo, gora, gamestate):
        if self.srodek[1] < gora.srodek[1] and self.srodek[0] < lewo.srodek[0]:
            self.pozycja = 'gl'
            gamestate[0][0] = 'x'
        if self.srodek[1] < gora.srodek[1] and self.srodek[0] > prawo.srodek[0]:
            self.pozycja = 'gp'
            gamestate[0][2] = 'x'
        if self.srodek[1] > dol.srodek[1] and self.srodek[0] > prawo.srodek[0]:
            self.pozycja = 'dp'
            gamestate[2][2] = 'x'
        if self.srodek[1] > dol.srodek[1] and self.srodek[0] < lewo.srodek[0]:
            self.pozycja = 'dl'
            gamestate[2][0] = 'x'
        if self.srodek[1] < gora.srodek[1] and self.srodek[0] > lewo.srodek[0] and self.srodek[0] < prawo.srodek[0]:
            self.pozycja = 'gs'
            gamestate[0][1] = 'x'
        if self.srodek[1] > dol.srodek[1] and self.srodek[0] > lewo.srodek[0] and self.srodek[0] < prawo.srodek[0]:
            self.pozycja = 'ds'
            gamestate[2][1] = 'x'
        if self.srodek[0] < lewo.srodek[0] and self.srodek[1] < dol.srodek[1] and self.srodek[1] > gora.srodek[1]:
            self.pozycja = 'sl'
            gamestate[1][0] = 'x'
        if self.srodek[0] > prawo.srodek[0] and self.srodek[1] < dol.srodek[1] and self.srodek[1] > gora.srodek[1]:
            self.pozycja = 'sp'
            gamestate[1][2] = 'x'
        if self.srodek[0] > lewo.srodek[0] and self.srodek[0] < prawo.srodek[0] and self.srodek[1] < dol.srodek[1] and \
                self.srodek[1] > gora.srodek[1]:
            self.pozycja = 'ss'
            gamestate[1][1] = 'x'


class Circle:
    def __init__(self, x=0, y=0, r=0):
        self.x = x
        self.y = y
        self.r = r
        self.pozycja = ''

    def okreslpozycje(self, prawo, dol, lewo, gora, gamestate):
        if self.y < gora.srodek[1] and self.x < lewo.srodek[0]:
            self.pozycja = 'gl'
            gamestate[0][0] = 'o'
        if self.y < gora.srodek[1] and self.x > prawo.srodek[0]:
            self.pozycja = 'gp'
            gamestate[0][2] = 'o'
        if self.y > dol.srodek[1] and self.x > prawo.srodek[0]:
            self.pozycja = 'dp'
            gamestate[2][2] = 'o'
        if self.y > dol.srodek[1] and self.x < lewo.srodek[0]:
            self.pozycja = 'dl'
            gamestate[2][0] = 'o'
        if self.y < gora.srodek[1] and self.x > lewo.srodek[0] and self.x < prawo.srodek[0]:
            self.pozycja = 'gs'
            gamestate[0][1] = 'o'
        if self.y > dol.srodek[1] and self.x > lewo.srodek[0] and self.x < prawo.srodek[0]:
            self.pozycja = 'ds'
            gamestate[2][1] = 'o'
        if self.x < lewo.srodek[0] and self.y < dol.srodek[1] and self.y > gora.srodek[1]:
            self.pozycja = 'sl'
            gamestate[1][0] = 'o'
        if self.x > prawo.srodek[0] and self.y < dol.srodek[1] and self.y > gora.srodek[1]:
            self.pozycja = 'sp'
            gamestate[1][2] = 'o'
        if self.x > lewo.srodek[0] and self.x < prawo.srodek[0] and self.y < dol.srodek[1] and self.y > gora.srodek[1]:
            self.pozycja = 'ss'
            gamestate[1][1] = 'o'


class Pole:
    def __init__(self, ydol=-1, ygora=-1, xlewo=-1, xprawo=-1):
        self.ydol = ydol
        self.ygora = ygora
        self.xlewo = xlewo
        self.xprawo = xprawo


def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)


def znajdzpola(prawo, dol, lewo, gora, pola):
    for pole in pola:
        pass


def znajdzkolka(obraz):
    kolka = []
    circles = cv2.HoughCircles(obraz, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=40, minRadius=1, maxRadius=100)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(lines_edges, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(lines_edges, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    if circles is not None:
        for i in circles:
            kolka.append(Circle(i[0], i[1], i[2]))
        for i in kolka:
            i.okreslpozycje(kreski[0], kreski[1], kreski[2], kreski[3], gamestate)
    return kolka


def znajdzkrzyze(obraz):
    krzyze = []
    for i in iksy:
        template = cv2.imread(i)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        template = cv2.Canny(template, 50, 200)
        # template = cv2.dilate(template,kernel,iterations = 1)
        (tH, tW) = template.shape[:2]
        found = None

        for scale in np.linspace(0.1, 3.0, 20)[::-1]:

            # Resize image to scale and keep track of ratio
            resized = maintain_aspect_ratio_resize(obraz, width=int(obraz.shape[1] * scale))
            r = obraz.shape[1] / float(resized.shape[1])

            # Stop if template image size is larger than resized image
            if resized.shape[0] < tH or resized.shape[1] < tW:
                break

            # Detect edges in resized image and apply template matching
            canny = cv2.Canny(resized, 50, 200)
            detected = cv2.matchTemplate(canny, template, cv2.TM_CCOEFF)
            (_, max_val, _, max_loc) = cv2.minMaxLoc(detected)

            '''
            clone = np.dstack([canny, canny, canny])
            cv2.rectangle(clone, (max_loc[0], max_loc[1]), (max_loc[0] + tW, max_loc[1] + tH), (0,255,0), 2)
            cv2.imshow('visualize', clone)
            cv2.waitKey(0)
            '''
            if found is None or max_val > found[0]:
                found = (max_val, max_loc, r)

        (_, max_loc, r) = found
        (start_x, start_y) = (int(max_loc[0] * r), int(max_loc[1] * r))
        (end_x, end_y) = (int((max_loc[0] + tW) * r), int((max_loc[1] + tH) * r))
        krzyze.append(cross(start_x, start_y, end_x, end_y))

        # Draw bounding box on ROI
        cv2.rectangle(lines_edges, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        for i in krzyze:
            i.policzsrodek()
            i.okreslpozycje(kreski[0], kreski[1], kreski[2], kreski[3], gamestate)

    return krzyze


def znajdzlinie(obraz):
    rho = 1
    theta = np.pi / 180
    threshold = 15
    min_line_length = 100
    max_line_gap = 20
    lines = cv2.HoughLinesP(obraz, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    linie = []
    for i in lines:
        linie.append(linia(i[0][0], i[0][1], i[0][2], i[0][3]))

    for i in linie:
        i.policzdlugosc()
        i.policzsrodek()
        i.sprawdzpozycje(img_width, img_height)
        i.sprawdzorientacje()
    return linie


def znajdzkrawedzie():
    pion = []
    poziom = []
    pionprawo = []
    pionlewo = []
    poziomgora = []
    poziomdol = []
    for i in linie:
        if i.orientacja == 1 and i.pozycjax >= 0.5 and i.pozycjax <= 0.9:
            pionprawo.append(i)
            pion.append(i)
        if i.orientacja == 1 and i.pozycjax < 0.5 and i.pozycjax > 0.1:
            pionlewo.append(i)
            pion.append(i)
        if i.orientacja == 0 and i.pozycjay < 0.5 and i.pozycjay > 0.1:
            poziomgora.append(i)
            poziom.append(i)
        if i.orientacja == 0 and i.pozycjay >= 0.5 and i.pozycjay <= 0.9:
            poziomdol.append(i)
            poziom.append(i)

    najdluzszapionprawo = pionprawo[0]
    for i in pion:
        if i.dlugosc > najdluzszapionprawo.dlugosc and i.pozycjax > 0.5:
            najdluzszapionprawo = i
    najdluzszapionlewo = pionlewo[0]
    for i in pion:
        if i.dlugosc > najdluzszapionlewo.dlugosc and i.pozycjax < 0.5:
            najdluzszapionlewo = i
    najdluzszapoziomgora = poziomgora[0]
    for i in poziom:
        if i.dlugosc > najdluzszapoziomgora.dlugosc and i.pozycjay < 0.5:
            najdluzszapoziomgora = i
    najdluzszapoziomdol = poziomdol[0]
    for i in poziom:
        if i.dlugosc > najdluzszapoziomdol.dlugosc and i.pozycjay > 0.5:
            najdluzszapoziomdol = i

    maksikx = pion[0]
    maksiky = poziom[0]
    minix = pion[0]
    miniy = poziom[0]
    dl = 30
    for i in pionprawo:
        if i.pozycjax > maksikx.pozycjax and i.dlugosc > najdluzszapionprawo.dlugosc - dl:
            maksikx = i
    for i in pionlewo:
        if i.pozycjax < minix.pozycjax and i.dlugosc > najdluzszapionlewo.dlugosc - dl:
            minix = i

    for i in poziomdol:
        if i.pozycjay > maksiky.pozycjay and i.dlugosc > najdluzszapoziomdol.dlugosc - dl:
            maksiky = i
    for i in poziomgora:
        if i.pozycjay < miniy.pozycjay and i.dlugosc > najdluzszapoziomgora.dlugosc - dl:
            miniy = i

    najdluzsze = [najdluzszapionlewo, najdluzszapionprawo, najdluzszapoziomdol, najdluzszapoziomgora]
    kreski = [maksikx, maksiky, minix, miniy]
    return kreski , najdluzsze


def pokazstangry():
    print("Stan gry to:")
    for line in gamestate:
        linetxt = ""
        for cel in line:
            linetxt = linetxt + "|" + cel
        linetxt = linetxt + '|'
        print(linetxt)
    if (gamestate.count('x') >= gamestate.count('o')):
        print("Teraz ruch gracza O")
    else:
        print("Teraz ruch gracza X")


def pokazplansze():
    cv2.imshow('image1', lines_edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
zdjecia = ['q2.png', 'q3.png', '10.png']
for i in zdjecia:
    # to sa wartosci testowe wykorzystywane do obliczen
    gamestate = [["-", "-", "-"], ["-", "-", "-"], ["-", "-", "-"]]
    kernel = np.ones((7, 7), np.uint8)
    img = cv2.imread('zdjecia/'+i)  # najlepsze do testow 10.png, 11.png, 2.jpg #odreczny rysunek q.png (nie lapie dobrze lini)
    img_width = img.shape[1]
    img_height = img.shape[0]
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img_g, 127, 255, cv2.THRESH_BINARY)
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(img_g, (kernel_size, kernel_size), 0)
    low_threshold = 50
    high_threshold = 100
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    erosion1 = edges
    erosion2 = edges
    for i in range(5):
        dilation1 = cv2.dilate(erosion1, kernel, iterations=1)
        dilation2 = cv2.dilate(erosion2, kernel, iterations=3)
        erosion1 = cv2.erode(dilation1, kernel, iterations=1)
        erosion2 = cv2.erode(dilation2, kernel, iterations=2)
    line_image = np.copy(img) * 0
    gotowe = np.copy(img) * 0

    # tu wykonywane sa obliczenia i konczy sie gra
    linie = znajdzlinie(erosion1)
    kreski, najdluzsze = znajdzkrawedzie()
    cv2.imshow('image1', erosion1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for i in kreski:
        cv2.line(gotowe, (i.x1, i.y1), (i.x2, i.y2), (255, 255, 0), 5)
    pola = []
    for i in range(9):
        pola.append(Pole())
    lines_edges = cv2.addWeighted(img, 0.8, gotowe, 1, 0)
    kolka = znajdzkolka(blur_gray)  # dziala tez dobrze na blur_gray
    krzyze = znajdzkrzyze(erosion1)
    pokazstangry()
    pokazplansze()