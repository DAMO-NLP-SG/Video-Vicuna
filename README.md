
<p align="center" width="100%">
<a target="_blank"><img src="figs/video_llama_logo.jpg" alt="Video-LLaMA" style="width: 50%; min-width: 200px; display: block; margin: auto;"></a>
</p>

# Video-LLaMA: Video Anlamada Eğitimli Görsel-İşitsel Dil Modeli
<!-- **Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding** -->

Bu, büyük dil modellerini video ve ses anlama yetenekleriyle güçlendirmeyi amaçlayan Video-LLaMA projesinin deposudur.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://modelscope.cn/studios/damo/video-llama/summary'><img src='https://img.shields.io/badge/ModelScope-Demo-blueviolet'></a>
<a href='https://www.modelscope.cn/models/damo/videollama_7b_llama2_finetuned/summary'><img src='https://img.shields.io/badge/ModelScope-Checkpoint-blueviolet'></a>
<a href='https://huggingface.co/spaces/DAMO-NLP-SG/Video-LLaMA'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<a href='https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-blue'></a> 
<a href='https://arxiv.org/abs/2306.02858'><img src='https://img.shields.io/badge/Paper-PDF-red'></a>
</div>

## Haberler
- <h3> [2024.06.03] 🚀🚀 Daha güçlü performans ve kullanımı daha kolay kod tabanı ile <a href='https://github.com/DAMO-NLP-SG/VideoLLaMA2'>VideoLLaMA2</a>'yi resmi olarak yayınlıyoruz, deneyin!</h3>
- [11.14] ⭐️ Mevcut README dosyası yalnızca **Video-LLaMA-2** (dil çözücü olarak LLaMA-2-Chat) içindir, önceki Video-LLaMA sürümünü (dil çözücü olarak Vicuna) kullanma talimatlarına [buradan](https://github.com/DAMO-NLP-SG/Video-LLaMA/blob/main/README_Vicuna.md) ulaşabilirsiniz.
- [08.03] 🚀🚀 Dil çözücü olarak [Llama-2-7B/13B-Chat](https://huggingface.co/meta-llama) kullanan **Video-LLaMA-2**'yi yayınladık
    - Artık delta ağırlıkları ve ayrı Q-former ağırlıkları YOK, Video-LLaMA'yı çalıştırmak için gereken tüm ağırlıklar burada :point_right: [[7B](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned)][[13B](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Finetuned)] 
    - Önceden eğitilmiş kontrol noktalarımızdan başlayarak daha fazla özelleştirmeye izin verir [[7B-Pretrained](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Pretrained)] [[13B-Pretrained](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Pretrained)]
- [06.14] **NOT**: Mevcut çevrimiçi interaktif demo öncelikle İngilizce sohbet içindir ve Vicuna/LLaMA Çince metinleri çok iyi temsil edemediğinden Çince sorular sormak için **İYİ** bir seçenek olmayabilir.
- [06.13] **NOT**: Şu anda diğer çözücüler için birkaç VL kontrol noktamız olmasına rağmen, ses desteği **YALNIZCA** Vicuna-7B içindir.
- [06.10] **NOT**: Tüm çerçeve (ses dalı ile) A10-24G'de normal olarak çalışamadığından HF demosunu henüz güncellemedik. Mevcut çalışan demo hala önceki Video-LLaMA sürümüdür. Bu sorunu yakında düzelteceğiz.
- [06.08] 🚀🚀 Ses destekli Video-LLaMA'nın kontrol noktalarını yayınladık. Dokümantasyon ve örnek çıktılar da güncellendi.    
- [05.22] 🚀🚀 İnteraktif demo çevrimiçi, Video-LLaMA'mızı (dil çözücü olarak **Vicuna-7B** ile) [Hugging Face](https://huggingface.co/spaces/DAMO-NLP-SG/Video-LLaMA) ve [ModelScope](https://pre.modelscope.cn/studios/damo/video-llama/summary)'da deneyin!!
- [05.22] ⭐️ Vicuna-7B ile oluşturulan **Video-LLaMA v2**'yi yayınladık
- [05.18] 🚀🚀 Çince video tabanlı sohbeti destekler 
    - [**Video-LLaMA-BiLLA**](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-Series/resolve/main/finetune-billa7b-zh.pth): Dil çözücü olarak [BiLLa-7B-SFT](https://huggingface.co/Neutralzz/BiLLa-7B-SFT)'yi tanıttık ve video-dil hizalı modeli (yani, aşama 1 modeli) makine çevirisi yapılmış [VideoChat](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) talimatları ile ince ayar yaptık.   
    - [**Video-LLaMA-Ziya**](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-Series/resolve/main/finetune-ziya13b-zh.pth): Video-LLaMA-BiLLA ile aynı ancak dil çözücü [Ziya-13B](https://huggingface.co/IDEA-CCNL/Ziya-LLaMA-13B-v1) olarak değiştirildi.    
- [05.18] ⭐️ Video-LLaMA'mızın tüm varyantlarının model ağırlıklarını saklamak için bir Hugging Face [deposu](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-Series) oluşturduk.
- [05.15] ⭐️ [**Video-LLaMA v2**](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-Series/resolve/main/finetune-vicuna13b-v2.pth)'yi yayınladık: Video-LLaMA'nın talimat izleme yeteneğini daha da geliştirmek için [VideoChat](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) tarafından sağlanan eğitim verilerini kullandık.
- [05.07] Önceden eğitilmiş ve talimat ayarlı kontrol noktaları dahil olmak üzere **Video-LLaMA**'nın ilk sürümünü yayınladık.

<p align="center" width="100%">
<a target="_blank"><img src="figs/architecture_v2.png" alt="Video-LLaMA" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

## Giriş

- Video-LLaMA, [BLIP-2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) ve [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) üzerine inşa edilmiştir. İki temel bileşenden oluşur: (1) Görüntü-Dil (VL) Dalı ve (2) Ses-Dil (AL) Dalı.
  - **VL Dalı** (Görsel kodlayıcı: ViT-G/14 + BLIP-2 Q-Former)
    - Video temsillerini hesaplamak için iki katmanlı bir video Q-Former ve bir kare gömme katmanı (her karenin gömülmelerine uygulanan) tanıtılmıştır. 
    - VL Dalını Webvid-2M video başlık veri setinde video-metin üretme görevi ile eğitiyoruz. Statik görsel kavramların anlaşılmasını geliştirmek için ön eğitim veri setine görüntü-metin çiftleri (~595K görüntü başlığı [LLaVA](https://github.com/haotian-liu/LLaVA)'dan) de ekledik.
    - Ön eğitimden sonra, VL Dalımızı [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA](https://github.com/haotian-liu/LLaVA) ve [VideoChat](https://github.com/OpenGVLab/Ask-Anything)'ten alınan talimat ayarlama verileri kullanarak daha da ince ayar yapıyoruz. 
  - **AL Dalı** (Ses kodlayıcı: ImageBind-Huge) 
    - Ses temsillerini hesaplamak için iki katmanlı bir ses Q-Former ve bir ses segment gömme katmanı (her ses segmentinin gömülmesine uygulanan) tanıtılmıştır.
    - Kullanılan ses kodlayıcı (yani ImageBind) zaten birden çok modalite arasında hizalanmış olduğundan, AL Dalını yalnızca ImageBind çıkışını dil çözücüye bağlamak için sadece video/görüntü talimat verileri üzerinde eğitiyoruz.    
- Çapraz modal eğitim sırasında yalnızca Video/Ses Q-Former, konumsal gömme katmanları ve doğrusal katmanlar eğitilebilir.

## Örnek Çıktılar

- **Arka plan sesli video**

<p float="left">
    <img src="https://github.com/DAMO-NLP-SG/Video-LLaMA/assets/18526640/7f7bddb2-5cf1-4cf4-bce3-3fa67974cbb3" style="width: 45%; margin: auto;">
    <img src="https://github.com/DAMO-NLP-SG/Video-LLaMA/assets/18526640/ec76be04-4aa9-4dde-bff2-0a232b8315e0" style="width: 45%; margin: auto;">
</p>

- **Ses efektsiz video**
<p float="left">
    <img src="https://github.com/DAMO-NLP-SG/Video-LLaMA/assets/18526640/539ea3cc-360d-4b2c-bf86-5505096df2f7" style="width: 45%; margin: auto;">
    <img src="https://github.com/DAMO-NLP-SG/Video-LLaMA/assets/18526640/7304ad6f-1009-46f1-aca4-7f861b636363" style="width: 45%; margin: auto;">
</p>

- **Statik görüntü**
<p float="left">
    <img src="https://github.com/DAMO-NLP-SG/Video-LLaMA/assets/18526640/a146c169-8693-4627-96e6-f885ca22791f" style="width: 45%; margin: auto;">
    <img src="https://github.com/DAMO-NLP-SG/Video-LLaMA/assets/18526640/66fc112d-e47e-4b66-b9bc-407f8d418b17" style="width: 45%; margin: auto;">
</p>

## Önceden Eğitilmiş & İnce Ayarlı Kontrol Noktaları

~~Aşağıdaki kontrol noktaları yalnızca öğrenilebilir parametreleri (konumsal gömme katmanları, Video/Ses Q-former ve doğrusal projeksiyon katmanları) saklar.~~

Aşağıdaki kontrol noktaları Video-LLaMA'yı başlatmak için tam ağırlıklardır (görsel kodlayıcı + ses kodlayıcı + Q-Former'lar + dil çözücü):

| Kontrol Noktası       | Bağlantı | Not |
|:------------------|-------------|-------------|
| Video-LLaMA-2-7B-Pretrained    | [bağlantı](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Finetuned/tree/main)       | WebVid (2.5M video-başlık çifti) ve LLaVA-CC3M (595k görüntü-başlık çifti) üzerinde önceden eğitilmiş |
| Video-LLaMA-2-7B-Finetuned | [bağlantı](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/tree/main) | [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4), [LLaVA](https://github.com/haotian-liu/LLaVA) ve [VideoChat](https://github.com/OpenGVLab/Ask-Anything)'ten alınan talimat ayarlama verileri üzerinde ince ayar yapılmış |

| Video-LLaMA-2-13B-Pretrained    | [bağlantı](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Pretrained/tree/main)       | 7B modeli ile aynı ancak daha büyük dil çözücü (LLaMA-2-13B-Chat) |
| Video-LLaMA-2-13B-Finetuned | [bağlantı](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-13B-Finetuned/tree/main) | 7B modeli ile aynı ancak daha büyük dil çözücü (LLaMA-2-13B-Chat) |

## Kurulum

1. Depoyu klonlayın ve bağımlılıkları yükleyin:
```bash
git clone https://github.com/DAMO-NLP-SG/Video-LLaMA.git
cd Video-LLaMA
pip install -r requirements.txt
```

2. Gerekli modellerle ilgili detaylara [buradan](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py) ulaşabilirsiniz:
- LLaMA-2-7B/13B-Chat
- İndirdiğiniz LLaMA-2 model dizinini `model_path`'de belirttiğiniz yere yerleştirin (bkz: aşağıdaki demo kodu). 

3. Eva ViT görsel kodlayıcıyı ve ImageBind ses kodlayıcıyı içeren [model ön eğitimi kaydını](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/tree/main) indirin.

4. Video içinden kare çıkarmak için ffmpeg kurulu olmalıdır, örneğin:  
```bash
# Ubuntu
apt update
apt install ffmpeg
# Windows (PowerShell Admin)
choco install ffmpeg
# MacOS
brew install ffmpeg
```

## Demo Başlatma

**Not**: Video-LLaMA'nın ses desteğinin **YALNIZCA** Vicuna-7B için olduğunu ve diğer dil çözücüleri için henüz mevcut olmadığını lütfen unutmayın.

1. Online Demo veya Local Demo arasında seçim yapın:

### Online Demo
- [ModelScope](https://modelscope.cn/studios/damo/video-llama/summary)
- [Hugging Face](https://huggingface.co/spaces/DAMO-NLP-SG/Video-LLaMA)

### Local Demo
**Not**: Ses desteğinin **YALNIZCA** Vicuna-7B için olduğunu ve diğer dil çözücüleri için henüz mevcut olmadığını lütfen unutmayın.
```python
import torch
from video_llama.constants import *
from video_llama.conversation import conv_templates, SeparatorStyle
from video_llama.model.builder import load_pretrained_model
from video_llama.utils.utils import disable_torch_init
from video_llama.processor import load_processor, process_images, process_video, process_audio

# ffmpeg yüklü olmalıdır
def initialize_model(model_path):
    disable_torch_init()
    
    model_name = "video_llama"
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    model = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_name,
        device=device,
        half=True,
        verbose=True
    )
    
    vis_processor = load_processor()
    return model, vis_processor, device

# Modeli yükle
model_path = "DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned" # veya "DAMO-NLP-SG/Video-LLaMA-2-13B-Finetuned"
model, vis_processor, device = initialize_model(model_path)

# Konuşmayı başlat
conv = conv_templates["v1"].copy()
conv.messages = []

# Herhangi bir video/görüntü/ses girdisi ile etkileşime geç
video_path = "val_video/1.mp4"
prompt = "Bu videoda ne oluyor?"

if video_path.endswith(('.mp4', '.avi', '.mov')):  # Video dosyası
    imgs, audio = process_video(video_path, vis_processor, device)
    audio_flag = True if audio is not None else False
    if len(imgs) > 0:  # Video kareleri başarıyla çıkarıldıysa
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "Anlayabilmem için biraz düşünmeme izin verin...")
        output = model.generate(imgs, audio if audio_flag else None, conv, temperature=0.7 if audio_flag else 0.2)
        conv.messages[-1][-1] = output
        print(f"Asistan: {output}")

elif video_path.endswith(('.jpg', '.png')):  # Görüntü dosyası
    image = process_images(video_path, vis_processor, device)
    if image is not None:  # Görüntü başarıyla yüklendiyse
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "Anlayabilmem için biraz düşünmeme izin verin...")
        output = model.generate(image, None, conv, temperature=0.2)
        conv.messages[-1][-1] = output
        print(f"Asistan: {output}")

else:
    print("Desteklenmeyen dosya formatı!")
```

## Sınırlamalar ve Sorumluluk Reddi

* Video-LLaMA zaman zaman halüsinasyonlar yaşayabilir, bu nedenle sonuçların doğruluğundan emin olunması önerilir.
* Video-LLaMA tehlikeli, yasa dışı, açık saçık, önyargılı ya da başka şekilde uygunsuz içerik üretmemeye çalışsa da beklenmedik sonuçlar üretebilir.
* Video-LLaMA'nın çıktıları yalnızca araştırma amaçlı kullanılmalıdır.
* İlk ısınma çıktısının kalitesi genellikle düşük olabilir.

## Atıfta Bulunma

```bibtex
@article{zhang2023videollama,
  title={Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding},
  author={Zhang, Hang and Li, Xin and Bing, Lidong},
  journal={arXiv preprint arXiv:2306.02858},
  year={2023}
}
```

## Lisans
- Video-LLaMA'nın kaynak kodu [Apache 2.0](LICENSE) lisansı altında yayınlanmıştır.
- Kullandığımız modeller ve veri setleri için lütfen ilgili lisanslara başvurun: [LLaMA](https://github.com/facebookresearch/llama), [ImageBind](https://github.com/facebookresearch/ImageBind), [Eva-CLIP](https://github.com/baaivision/EVA/tree/master/EVA-CLIP).

## Teşekkürler
- [BLIP-2](https://huggingface.co/docs/transformers/main/model_doc/blip-2) video-dil ön eğitimi için temel almış olduğumuz görüntü-dil modelidir.
- [ImageBind](https://github.com/facebookresearch/ImageBind) ses modalitesini diğer modalitelerle hizalamak için kullanılmıştır.
- [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) temel etkileşim kodu ve demo UI için temel oluşturmuştur.
- [LLaMA](https://github.com/facebookresearch/llama) ve [Vicuna](https://github.com/lm-sys/FastChat) dil çözücülerimiz için temel modellerdir.
- [🤗 Hugging Face](https://github.com/huggingface) tüm modeller ve uygulamaları barındırdığı için kullanılmıştır.