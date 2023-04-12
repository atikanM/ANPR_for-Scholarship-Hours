import cv2
import easyocr
from matplotlib import pyplot as plt
import numpy as np
import imutils
from PIL import ImageFont, ImageDraw, Image 
from thefuzz import fuzz, process


def closest_match_province(test_province):

    provinces = ['กระบี่', 'กรุงเทพมหานคร', 'กาญจนบุรี', 'กาฬสินธุ์', 'กำแพงเพชร', 'ขอนแก่น', 'จันทบุรี', 'ฉะเชิงเทรา', 'ชลบุรี', 'ชัยนาท', 'ชัยภูมิ', 'ชุมพร', 'เชียงราย', 'เชียงใหม่', 'ตรัง', 'ตราด', 'ตาก', 'นครนายก'
    , 'นครปฐม', 'นครพนม', 'นครราชสีมา', 'นครศรีธรรมราช', 'นครสวรรค์', 'นนทบุรี', 'นราธิวาส', 'น่าน', 'บึงกาฬ', 'บุรีรัมย์', 'ปทุมธานี', 'ประจวบคีรีขันธ์', 'ปราจีนบุรี', 'ปัตตานี', 'พระนครศรีอยุธยา', 'พะเยา', 'พังงา', 
    'พัทลุง', 'พิจิตร', 'พิษณุโลก', 'เพชรบุรี', 'เพชรบูรณ์', 'แพร่', 'ภูเก็ต', 'มหาสารคาม', 'มุกดาหาร', 'แม่ฮ่องสอน', 'ยโสธร', 'ยะลา', 'ร้อยเอ็ด', 'ระนอง', 'ระยอง', 'ราชบุรี', 'ลพบุรี', 'ลำปาง'
    , 'ลำพูน', 'เลย', 'ศรีสะเกษ', 'สกลนคร', 'สงขลา', 'สตูล', 'สมุทรปราการ', 'สมุทรสงคราม', 'สมุทรสาคร', 'สระแก้ว', 'สระบุรี', 'สิงห์บุรี', 'สุโขทัย', 'สุพรรณบุรี', 'สุราษฎร์ธานี', 'สุรินทร์', 'หนองคาย', 
    'หนองบัวลำภู', 'อ่างทอง', 'อำนาจเจริญ', 'อุดรธานี', 'อุตรดิตถ์', 'อุทัยธานี', 'อุบลราชธานี', 'เบตง']

    similarities = []

    for province in provinces:
        similarities.append(fuzz.ratio(test_province, province))

    max_similarity = max(similarities)
    for index, similarity in enumerate(similarities):
        if similarity == max_similarity:
            predicted_province = provinces[index]
            break

    return predicted_province

# PART 1: Read image, Apply Grayscale

img = cv2.imread('bad_no contour_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)) #Matplotlib expect a RGB-type to plot (our is BGR)
# plt.show() 

##################################################################

# PART 2: Apply filter and Edge dectection

blurfilter = cv2.bilateralFilter(gray, 11, 17, 17) 
edged = cv2.Canny(blurfilter, 30, 200) 
# plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
# plt.show() 

##################################################################

# PART 3: Find contours (SHAPE : 4 points == rectangle)

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20] 

##################################################################

#PART 4: Apply mask on to the contours (plate)

location = None
counter = 0

def applymask():
    global mask,new_image
    mask = np.zeros(gray.shape, np.uint8) 
    new_image = cv2.drawContours(mask, [location], 0, 255, -1) 
    new_image = cv2.bitwise_and(img, img, mask=mask) 
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.show()

###################################################################

#PART 5: Extract only the masked area

def extractplate():
    global x,y,x1,y1,x2,y2,cropped_image
    (x, y) = np.where(mask==255) 
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y)) 
    cropped_image = gray[x1:x2+1, y1:y2+1] 
    # plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    # plt.show() 


def findplate():
    global location,i,approx,counter
    for i in contours:
        approx = cv2.approxPolyDP(i, 10, True) 
        counter += 1
        if len(approx) == 4:
            location = approx
            i = i
            break
    applymask()
    extractplate()
    #print(location) #check if the location is correct
findplate()

##################################################################

# PART 6: Use EASYocr to read text

reader = easyocr.Reader(['th'])
result = reader.readtext(cropped_image)

if(len(result) == 0 or len(result) == 1):
    del contours[:counter]
    findplate()
#print(result)

# ##################################################################

# # PART 7: Overlay the result onto original image

# if(len(result) == 2): #for 3-4 (long) numbers plate
#     text_number = result[0][-2]
#     text_province = result[1][-2]
#     print(text_number)
    

# elif (len(result) == 3): #for 1-2 (short) numbers plate
#     text_number1 = result[0][-2]
#     text_number2 = result[1][-2]
#     text_province = result[2][-2]
#     text_number = text_number1 + " " + text_number2
#     print(text_number)


# text_province_adjust = closest_match_province(text_province)
# print(text_province_adjust)





# fontpath = "Norasi Bold.ttf" #import font
# font = ImageFont.truetype(fontpath, 50) #set font + size
# img_pil = Image.fromarray(img) #import image as an array
# draw = ImageDraw.Draw(img_pil) #set image to draw text
# draw.text((0, 0), text_number, font = font, fill = (0, 255, 0, 255))
# draw.text((0, 40), text_province_adjust, font = font, fill = (0, 255, 0, 255))
# img = np.array(img_pil) #store the modified(drew) image as an array

# result_final = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0), 3) #(image, top left corner, bottom right corner, color, thickness)
# plt.imshow(cv2.cvtColor(result_final, cv2.COLOR_BGR2RGB))
# plt.show() 
