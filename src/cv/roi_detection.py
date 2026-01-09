from torchvision.models.segmentation import deeplabv3_resnet50
from PIL import Image
import openpifpaf
import cv2
import torchvision.transforms as tvT
import torch

def hardcoded_roi(camera):
    if camera=="gray":
        if region=="abdomen":
            return (1200, 2000, 1100, 800)
        else:
            return (1200, 1200, 1100, 800)
    elif camera=="olympus":
        if region=="abdomen":
            return (750, 775, 500, 275)
        else:
            return (750, 500, 500, 275)
    elif camera=="flir":
        if region=="abdomen":
            return (200, 375, 225, 125)
        else:
            return (200, 250, 225, 125)
    else:
        if region=="abdomen":
             return (350, 1450, 300, 225)
        else:
            return (350, 1175, 300, 225)

def detect_ROI(image_path, camera, region="chest", image_type='RGB', conf_thr=0.1):
    pil_im = Image.open(image_path).convert(image_type)
    predictor = get_pifpaf_predictor()
    predictions, _, _ = predictor.pil_image(pil_im)
    W, H = pil_im.size

    if region == "all":
        return (0, 0, W, H)
    elif predictions is None or len(predictions) == 0:
        return hardcoded_roi(camera)

    person = predictions[0]
    key_points = person.data
    ls, rs = key_points[5], key_points[6]

    if ls[2] < conf_thr or rs[2] < conf_thr:
        return hardcoded_roi(camera)
                
    else:
        shoulder_y = (ls[1] + rs[1]) / 2.0
        x_left, x_right = float(min(ls[0], rs[0])), float(max(ls[0], rs[0]))
        width = max(1.0, x_right - x_left)
        left, right = x_left, x_right 
        top, bottom = shoulder_y, shoulder_y + 1.2 * width

    left, right = int(max(0, min(left, W-1))), int(max(left+1, min(right, W)))
    top, bottom = int(max(0, min(top, H-1))), int(max(top+1, min(bottom, H)))
    
    h_box = bottom - top
    half_h = int(max(1, h_box // 2))
    w_box = right - left

    if region == "chest": return (left, top, w_box, half_h)
    elif region == "abdomen": return (left, top + half_h, w_box, half_h)
    return (left, top, w_box, half_h)

def segment_person_deeplab(frame_bgr):
    model = get_deeplab_model()
    transform = tvT.Compose([tvT.ToTensor(), tvT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad(): out = model(inp)["out"]
    labels = out.argmax(1).squeeze().cpu().numpy()
    mask = (labels == 15).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask
