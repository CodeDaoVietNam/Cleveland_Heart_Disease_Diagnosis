import easyocr
import re
import warnings

# Tắt cảnh báo đỏ của Torch trên Terminal Streamlit
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")

reader = None

def get_easyocr_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en', 'vi'], gpu=False, verbose=False)
    return reader

def extract_medical_data_from_image(image_bytes):
    reader = get_easyocr_reader()
    
    result = reader.readtext(image_bytes)
    text_list = [res[1].lower() for res in result]
    full_text = " ".join(text_list)
    
    extracted_data = {}
    
    chol_match = re.search(r'cholesterol\s*[-:=]?\s*(\d{2,3})', full_text)
    if not chol_match:
        chol_match = re.search(r'chol\s*[-:=]?\s*(\d{2,3})', full_text)
        
    if chol_match:
        extracted_data['chol'] = int(chol_match.group(1))
        
    bp_match = re.search(r'(huyet ap|huyết áp|blood pressure|bp)\s*[-:=]?\s*(\d{2,3})', full_text)
    if bp_match:
        extracted_data['trestbps'] = int(bp_match.group(2))
        
    if 'fasting blood sugar' in full_text and '> 120' in full_text:
        extracted_data['fbs'] = 1.0
    elif 'fasting blood sugar':
        try:
             fbs_match = re.search(r'fasting blood sugar\s*[-:=]?\s*(\d{2,3})', full_text)
             if fbs_match and int(fbs_match.group(1)) > 120:
                 extracted_data['fbs'] = 1.0
             else:
                 extracted_data['fbs'] = 0.0
        except:
             pass

    return extracted_data, full_text
