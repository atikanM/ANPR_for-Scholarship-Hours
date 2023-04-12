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

img = cv2.imread('bad_wrong_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Change BGR to GRAY
# plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)) #Matplotlib expect a RGB-type to plot (our is BGR)
# plt.show() 

##################################################################

# PART 2: Apply filter and Edge dectection

blurfilter = cv2.bilateralFilter(gray, 11, 17, 17) #noise reduction : (source_image, diameter of pixel, sigmaColor, sigmaSpace)
edged = cv2.Canny(blurfilter, 30, 200) #edge detection : (source_image, thresholdValue 1, thresholdValue 2)
# plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
# plt.show() 

##################################################################

# PART 3: Find contours (SHAPE : 4 points == rectangle)

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #return result as TREE, return an approximate (start and end points - not every single point)
contours = imutils.grab_contours(keypoints) #simplify our contours (for sorting)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] #sort (descending) and return TOP 10 contours base on contourArea

location = None
for i in contours: #loop through our contours 
    approx = cv2.approxPolyDP(i, 10, True) #approximate polygon from each of the contour, specify how accurate/fine grain approximation (in this case = 10) 
    if len(approx) == 4: #if we have 4 keypoints, it is likely gonna be the plate 
        location = approx
        break
# print(location) 

##################################################################

# PART 4: Apply mask on to the contours (plate)

mask = np.zeros(gray.shape, np.uint8) #create blank mask with the shape similar to the original GRAYimage(xxx.jpg), fill it in with blank zeros
new_image = cv2.drawContours(mask, [location], 0, (255,255,255), thickness=-1) #draw a frame onto the blank mask (in this case = location) 
new_image = cv2.bitwise_and(img, img, mask=mask) #combine original image + modified mask --> return the segment of number plate
# plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
# plt.show()

###################################################################

#PART 5: Extract only the masked area

(x, y) = np.where(mask==255) #find every section that image isn't black (in form of SET of Coordinates (x,y))
(x1, y1) = (np.min(x), np.min(y)) #grab minx , miny (top left corner)
(x2, y2) = (np.max(x), np.max(y)) #grab maxx , maxy (bottom right corner)
cropped_image = gray[x1:x2+1, y1:y2+1] #filter by grabbing point x1 to x2+1 (x-axis) and point y1 to y2+1 (y-axis)
# plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
# plt.show() 

##################################################################

# PART 6: Use EASYocr to read text

reader = easyocr.Reader(['th'])
result = reader.readtext(cropped_image)
print(result)

##################################################################

# PART 7: Overlay the result onto original image

if(len(result) == 2): #for 3-4 (long) numbers plate
    text_number = result[0][-2]
    text_province = result[1][-2]
    print(text_number)
    

elif (len(result) == 3): #for 1-2 (short) numbers plate
    text_number1 = result[0][-2]
    text_number2 = result[1][-2]
    text_province = result[2][-2]
    text_number = text_number1 + " " + text_number2
    print(text_number)

text_province_adjust = closest_match_province(text_province)
print(text_province_adjust)

# text_numbersplit = text_number.split()
# print(text_numbersplit[0])
# print(text_numbersplit[1])
# print(text_province)

# font = cv2.FONT_HERSHEY_SIMPLEX
# result_number = cv2.putText(img, text=text_number, org=(approx[0][0][0], approx[1][0][1]+30), fontFace=font, fontScale=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA) #display number
# result_province = cv2.putText(img, text=text_province_adjust, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA) #display province
# result_final = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (255,0,0), 3)
# plt.imshow(cv2.cvtColor(result_final, cv2.COLOR_BGR2RGB))
# plt.show() 

fontpath = "Norasi Bold.ttf" #import font
font = ImageFont.truetype(fontpath, 50) #set font + size
img_pil = Image.fromarray(img) #import image
draw = ImageDraw.Draw(img_pil) #set image to draw text
draw.text((0, 0), text_number, font = font, fill = (0, 255, 0, 255))
draw.text((0, 40), text_province_adjust, font = font, fill = (0, 255, 0, 255))
img = np.array(img_pil) #store the modified(drew) image as an array

result_final = cv2.rectangle(img, tuple(location[0][0]), tuple(location[2][0]), (0,255,0), 3) #(image, top left corner, bottom right corner, color, thickness)
plt.imshow(cv2.cvtColor(result_final, cv2.COLOR_BGR2RGB))
plt.show() 
