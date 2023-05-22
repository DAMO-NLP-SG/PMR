#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  {
    "context": "Xinhua News Agency , Shanghai , August 31st , by reporter Jierong Zhou",
    "end_position": [
      2
    ],
    "entity_label": "ORG",
    "impossible": false,
    "qas_id": "0.2",
    "query": "organization entities are limited to companies, corporations, agencies, institutions and other groups of people.",
    "span_position": [
      "0;2"
    ],
    "start_position": [
      0
    ]
  }
"""

import os
import json

def normalize_word(word, language='english'):
    if language == "arabic":
        word = word[:word.find("#")]
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

def get_original_token(token):
    escape_to_original = {
        "-LRB-": "(",
        "-RRB-": ")",
        "-LSB-": "[",
        "-RSB-": "]",
        "-LCB-": "{",
        "-RCB-": "}",
    }
    if token in escape_to_original:
        token = escape_to_original[token]
    return token

def read_conll(input_file, delimiter=" "):
    """load ner dataset from CoNLL-format files."""
    dataset_item_lst = []
    with open(input_file, "r", encoding="utf-8") as r_f:
        datalines = r_f.readlines()

    cached_token, cached_label = [], []
    for idx, data_line in enumerate(datalines):
        data_line = data_line.strip()
        if data_line.startswith("-DOCSTART-"):
            continue
        if idx != 0 and len(data_line) == 0:
            dataset_item_lst.append([cached_token, cached_label])
            cached_token, cached_label = [], []
        else:
            token_label = data_line.split(delimiter)
            token_data_line, label_data_line = token_label[0], token_label[1]
            if label_data_line.startswith("M"):
                label_data_line = "I" + label_data_line[1:]
            elif label_data_line.startswith("E"):
                label_data_line = "I" + label_data_line[1:]
            elif label_data_line.startswith("S"):
                label_data_line = "B" + label_data_line[1:]
            token_data_line = get_original_token(token_data_line)
            token_data_line = normalize_word(token_data_line)
            cached_token.append(token_data_line)
            cached_label.append(label_data_line)
    new_data = []
    for x in dataset_item_lst:
        if x[0] != []:
            new_data.append(x)
    return new_data

def save_BIO(conll_f, save_addr):
    conll_f
    with open(save_addr, 'w') as writer:
        for i_s, sen in enumerate(conll_f):
            for i_w, w in enumerate(sen[0]):
                line = " ".join([sen[0][i_w], sen[1][i_w]])
                writer.write(line + '\n')
            writer.write('\n')

def conll2mrc(conll_f, lang):
    count = 0
    # label_details = {
    #     'ORG': "Organization entities are limited to named corporate, governmental, or other organizational entities.",
    #     'PER': "Person entities are named persons or family.",
    #     'LOC': "Location entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc.",
    #     }
    # from googletrans import Translator
    # translator = Translator(service_urls=['translate.google.cn'])
    # labels_all = {}
    # for lang in ['af', 'ar', 'bg', 'bn','el','et', 'eu', 'fa', 'fi', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'jv', 'ka'
    #              ,'kk','ko','ml','mr','ms','my','pt','ru','sw','ta','te','th','tl','tr','ur','vi','yo','zh']:
    #     label_details_one = {}
    #     if lang=='zh':
    #         lang = 'zh-CN'
    #     for label in label_details:
    #         result = translator.translate(label_details[label], dest=lang).text
    #         label_details_one[label] = result
    #     labels_all[lang] = label_details_one
    if lang == "en":
        label_details = {'ORG': "Organization entities are limited to named corporate, governmental, or other organizational entities.",
                         'PER': "Person entities are named persons or family.",
                         'LOC': "Location entities are the name of politically or geographically defined locations such as cities, provinces, countries, international regions, bodies of water, mountains, etc.",
                         }
    elif lang == "nl":
        label_details = {
            'ORG': "Organisatie-entiteiten zijn beperkt tot genoemde bedrijfs-, overheids- of andere organisatorische entiteiten.",
            'PER': "Persoon entiteiten zijn genoemde personen of familie.",
            'LOC': "Locatie-entiteiten zijn de naam van politiek of geografisch gedefinieerde locaties zoals steden, provincies, landen, internationale regio's, waterlichamen, bergen, enz.",
            }
    elif lang == "de":
        label_details = {
            'ORG': "Organisationseinheiten sind auf benannte Unternehmens-, Regierungs- oder andere Organisationseinheiten beschränkt.",
            'PER': "Personeneinheiten sind benannte Personen oder Familien.",
            'LOC': "Ortseinheiten sind die Namen politisch oder geografisch definierter Orte wie Städte, Provinzen, Länder, internationale Regionen, Gewässer, Berge usw.",
            }
    elif lang == "es":
        label_details = {'ORG': "Las entidades organizativas se limitan a entidades corporativas, gubernamentales u otras entidades organizativas nombradas.",
                         'PER': "Las entidades persona son personas nombradas o familia.",
                         'LOC': "Las entidades de ubicación son el nombre de ubicaciones definidas política o geográficamente, como ciudades, provincias, países, regiones internacionales, masas de agua, montañas, etc.",
                         }
    elif lang == "vi":
        label_details = {'ORG': "Các thực thể tổ chức được giới hạn trong các thực thể công ty, chính phủ hoặc các tổ chức khác được đặt tên.",
                         'PER': "Các thực thể cá nhân là những người được đặt tên hoặc gia đình.",
                         'LOC': "Thực thể vị trí là tên của các vị trí được xác định về mặt chính trị hoặc địa lý như thành phố, tỉnh, quốc gia, khu vực quốc tế, vùng nước, núi, v.v.",
                         }
    elif lang == "fr":
        label_details = {'ORG': "Les entités organisationnelles sont limitées aux entités corporatives, gouvernementales ou autres entités organisationnelles nommées.",
                         'PER': "Les entités de personne sont des personnes ou une famille nommées.",
                         'LOC': "Les entités de localisation sont le nom de lieux politiquement ou géographiquement définis tels que des villes, des provinces, des pays, des régions internationales, des plans d'eau, des montagnes, etc.",
                         }
    elif lang == "it":
        label_details = {'ORG': "Le entità dell'organizzazione sono limitate a entità aziendali, governative o di altro tipo nominate.",
                         'PER': "Le entità persona sono persone nominate o familiari.",
                         'LOC': "Le entità di localizzazione sono il nome di località definite politicamente o geograficamente come città, province, paesi, regioni internazionali, specchi d'acqua, montagne, ecc.",
                         }
    elif lang == "ja":
        label_details = {'ORG': "組織エンティティは、指定された企業、政府、またはその他の組織エンティティに限定されます。",
                         'PER': "個人エンティティは、名前付きの個人または家族です。",
                         'LOC': "ロケーション エンティティは、都市、州、国、国際地域、水域、山など、政治的または地理的に定義された場所の名前です。",
                         }
    elif lang == "ko":
        label_details = {'ORG': "조직 엔터티는 명명된 기업, 정부 또는 기타 조직 엔터티로 제한됩니다.",
                         'PER': "개인 엔티티는 이름이 지정된 개인 또는 가족입니다.",
                         'LOC': "위치 엔티티는 도시, 지방, 국가, 국제 지역, 수역, 산 등과 같이 정치적으로 또는 지리적으로 정의된 위치의 이름입니다.",
                         }
    elif lang == "zh":
        label_details = {'ORG': "组织实体仅限于命名的公司、政府或其他组织实体。",
                         'PER': "个人实体被命名为个人或家庭。",
                         'LOC': "位置实体是政治或地理上定义的位置的名称，例如城市、省、国家、国际区域、水体、山脉等。",
                         }
    elif lang == "af":
        label_details = {'ORG': "Organisasie-entiteite is beperk tot genoemde korporatiewe, regerings- of ander organisatoriese entiteite.",
                         'PER': "Persoonsentiteite is genoemde persone of familie.",
                         'LOC': "Liggingsentiteite is die naam van polities of geografies gedefinieerde liggings soos stede, provinsies, lande, internasionale streke, waterliggame, berge, ens.",
                         }
    elif lang == "ar":
        label_details = {'ORG': "تقتصر كيانات المؤسسة على كيانات مؤسسية أو حكومية أو كيانات تنظيمية أخرى مسماة.",
                         'PER': "يتم تسمية كيانات الشخص كأشخاص أو عائلة.",
                         'LOC': "كيانات الموقع هي اسم المواقع المحددة سياسياً أو جغرافياً مثل المدن ، والمقاطعات ، والبلدان ، والمناطق الدولية ، والمسطحات المائية ، والجبال ، إلخ.",
                         }
    elif lang == "bg":
        label_details = {'ORG': "Организационните единици са ограничени до наименувани корпоративни, правителствени или други организационни единици.",
                         'PER': "Личните субекти са именувани лица или семейство.",
                         'LOC': "Обектите на местоположението са имената на политически или географски определени местоположения като градове, провинции, държави, международни региони, водни обекти, планини и др.",
                         }
    elif lang == "bn":
        label_details = {'ORG': "সংস্থার সত্তাগুলি নামযুক্ত কর্পোরেট, সরকারী বা অন্যান্য সাংগঠনিক সত্তার মধ্যে সীমাবদ্ধ।",
                         'PER': "ব্যক্তি সত্তার নাম ব্যক্তি বা পরিবার।",
                         'LOC': "অবস্থান সত্তা হল রাজনৈতিকভাবে বা ভৌগলিকভাবে সংজ্ঞায়িত অবস্থানের নাম যেমন শহর, প্রদেশ, দেশ, আন্তর্জাতিক অঞ্চল, জলাশয়, পর্বত ইত্যাদি।",
                         }
    elif lang == "el":
        label_details = {'ORG': "Οι οντότητες του οργανισμού περιορίζονται σε επώνυμες εταιρικές, κυβερνητικές ή άλλες οργανωτικές οντότητες.",
                         'PER': "Οι οντότητες προσώπων ονομάζονται πρόσωπα ή οικογένεια.",
                         'LOC': "Οι οντότητες τοποθεσίας είναι το όνομα πολιτικά ή γεωγραφικά καθορισμένων τοποθεσιών όπως πόλεις, επαρχίες, χώρες, διεθνείς περιοχές, υδάτινα σώματα, βουνά κ.λπ.",
                         }
    elif lang == "et":
        label_details = {'ORG': "Organisatsiooni üksused piirduvad nimetatud ettevõtte, valitsusasutuste või muude organisatsiooniliste üksustega.",
                         'PER': "Isiku entiteete nimetatakse isikuteks või perekondadeks.",
                         'LOC': "Asukohaüksused on poliitiliselt või geograafiliselt määratletud asukohtade (nt linnad, provintsid, riigid, rahvusvahelised piirkonnad, veekogud, mäed jne) nimed.",
                         }
    elif lang == "eu":
        label_details = {'ORG': "Antolakuntza-entitateak izendatutako korporazio, gobernu edo bestelako erakunde entitateetara mugatzen dira.",
                         'PER': "Pertsona-entitateak pertsona edo familia izendatuak dira.",
                         'LOC': "Kokapen-entitateak politikoki edo geografikoki definitutako kokapenen izenak dira, hala nola hiriak, probintziak, herrialdeak, nazioarteko eskualdeak, ur-masak, mendiak, etab.",
                         }
    elif lang == "fa":
        label_details = {'ORG': "نهادهای سازمانی محدود به نهادهای سازمانی، دولتی یا سازمانی دیگر هستند.",
                         'PER': "نهادهای شخصی افراد یا خانواده نامیده می شوند.",
                         'LOC': "نهادهای مکان نام مکان هایی هستند که از نظر سیاسی یا جغرافیایی تعریف شده اند مانند شهرها، استان ها، کشورها، مناطق بین المللی، توده های آبی، کوه ها و غیره.",
                         }
    elif lang == "fi":
        label_details = {'ORG': "Organisaatiokokonaisuudet rajoittuvat nimettyihin yritys-, hallinto- tai muihin organisaatiokokonaisuuksiin.",
                         'PER': "Henkilökokonaisuudet ovat nimettyjä henkilöitä tai perheitä.",
                         'LOC': "Sijaintientiteetit ovat poliittisesti tai maantieteellisesti määriteltyjen paikkojen nimiä, kuten kaupunkeja, provinsseja, maita, kansainvälisiä alueita, vesistöjä, vuoria jne.",
                         }
    elif lang == "he":
        label_details = {'ORG': "ישויות ארגוניות מוגבלות לגופים תאגידיים, ממשלתיים או ארגוניים אחרים בעלי שם.",
                         'PER': "ישויות אישיות הן אנשים או משפחה בשם.",
                         'LOC': "ישויות מיקום הן שמם של מיקומים מוגדרים פוליטית או גיאוגרפית כגון ערים, מחוזות, מדינות, אזורים בינלאומיים, גופי מים, הרים וכו'.",
                         }
    elif lang == "hi":
        label_details = {'ORG': "संगठन निकाय नामित कॉर्पोरेट, सरकारी, या अन्य संगठनात्मक संस्थाओं तक सीमित हैं।",
                         'PER': "व्यक्ति संस्थाओं का नाम व्यक्ति या परिवार है।",
                         'LOC': "स्थान निकाय राजनीतिक या भौगोलिक रूप से परिभाषित स्थानों जैसे शहरों, प्रांतों, देशों, अंतर्राष्ट्रीय क्षेत्रों, जल निकायों, पहाड़ों आदि के नाम हैं।",
                         }
    elif lang == "hu":
        label_details = {'ORG': "A szervezeti entitások a megnevezett vállalati, kormányzati vagy egyéb szervezeti entitásokra korlátozódnak.",
                         'PER': "A személy entitások megnevezett személyek vagy családok.",
                         'LOC': "A hely entitások politikailag vagy földrajzilag meghatározott helyek nevei, például városok, tartományok, országok, nemzetközi régiók, víztestek, hegyek stb.",
                         }
    elif lang == "id":
        label_details = {'ORG': "Entitas organisasi terbatas pada nama perusahaan, pemerintah, atau entitas organisasi lainnya.",
                         'PER': "Entitas orang disebut orang atau keluarga.",
                         'LOC': "Entitas lokasi adalah nama lokasi yang ditentukan secara politis atau geografis seperti kota, provinsi, negara, wilayah internasional, badan air, pegunungan, dll.",
                         }
    elif lang == "jv":
        label_details = {'ORG': "Entitas organisasi diwatesi mung perusahaan, pemerintah, utawa entitas organisasi liyane.",
                         'PER': "Entitas wong diarani wong utawa kulawarga.",
                         'LOC': "Entitas lokasi yaiku jeneng lokasi sing ditemtokake sacara politis utawa geografis kayata kutha, provinsi, negara, wilayah internasional, badan banyu, gunung, lsp.",
                         }
    elif lang == "ka":
        label_details = {'ORG': "ორგანიზაციის ერთეულები შემოიფარგლება დასახელებული კორპორატიული, სამთავრობო ან სხვა ორგანიზაციული ერთეულებით.",
                         'PER': "პირთა სუბიექტები დასახელებულია პირები ან ოჯახი.",
                         'LOC': "მდებარეობის ერთეულები არის პოლიტიკურად ან გეოგრაფიულად განსაზღვრული მდებარეობების სახელები, როგორიცაა ქალაქები, პროვინციები, ქვეყნები, საერთაშორისო რეგიონები, წყლის ობიექტები, მთები და ა.შ.",
                         }
    elif lang == "kk":
        label_details = {'ORG': "Ұйым нысандары аталған корпоративтік, үкіметтік немесе басқа ұйымдық құрылымдармен шектеледі.",
                         'PER': "Тұлға субъектілері - аталған тұлғалар немесе отбасы.",
                         'LOC': "Орналасу нысандары - қалалар, провинциялар, елдер, халықаралық аймақтар, су айдындары, таулар және т.б. сияқты саяси немесе географиялық тұрғыдан анықталған орындардың атауы.",
                         }
    elif lang == "ml":
        label_details = {'ORG': "ഓർഗനൈസേഷൻ എന്റിറ്റികൾ പേരുള്ള കോർപ്പറേറ്റ്, ഗവൺമെന്റൽ അല്ലെങ്കിൽ മറ്റ് ഓർഗനൈസേഷണൽ എന്റിറ്റികൾക്ക് മാത്രമായി പരിമിതപ്പെടുത്തിയിരിക്കുന്നു.",
                         'PER': "വ്യക്തി എന്റിറ്റികളെ വ്യക്തികൾ അല്ലെങ്കിൽ കുടുംബം എന്ന് വിളിക്കുന്നു.",
                         'LOC': "നഗരങ്ങൾ, പ്രവിശ്യകൾ, രാജ്യങ്ങൾ, അന്തർദേശീയ പ്രദേശങ്ങൾ, ജലാശയങ്ങൾ, പർവതങ്ങൾ മുതലായവ പോലുള്ള രാഷ്ട്രീയമായോ ഭൂമിശാസ്ത്രപരമായോ നിർവചിക്കപ്പെട്ട സ്ഥലങ്ങളുടെ പേരാണ് ലൊക്കേഷൻ എന്റിറ്റികൾ.",
                         }
    elif lang == "mr":
        label_details = {'ORG': "संस्था संस्था नामांकित कॉर्पोरेट, सरकारी किंवा इतर संस्थात्मक घटकांपुरती मर्यादित आहेत.",
                         'PER': "व्यक्ती संस्थांना व्यक्ती किंवा कुटुंब असे नाव दिले जाते.",
                         'LOC': "स्थान संस्था म्हणजे शहरे, प्रांत, देश, आंतरराष्ट्रीय प्रदेश, पाण्याचे स्रोत, पर्वत इ. यासारख्या राजकीय किंवा भौगोलिकदृष्ट्या परिभाषित स्थानांचे नाव.",
                         }
    elif lang == "ms":
        label_details = {'ORG': "Entiti organisasi terhad kepada entiti korporat, kerajaan atau organisasi lain yang dinamakan.",
                         'PER': "Entiti orang dinamakan orang atau keluarga.",
                         'LOC': "Entiti lokasi ialah nama lokasi yang ditentukan secara politik atau geografi seperti bandar, wilayah, negara, wilayah antarabangsa, badan air, gunung, dsb.",
                         }
    elif lang == "my":
        label_details = {'ORG': "အဖွဲ့အစည်းဆိုင်ရာ အဖွဲ့အစည်းများကို အမည်ပေးထားသည့် ကော်ပိုရိတ်၊ အစိုးရ သို့မဟုတ် အခြားအဖွဲ့အစည်းဆိုင်ရာ အဖွဲ့အစည်းများအတွက် ကန့်သတ်ထားသည်။",
                         'PER': "လူပုဂ္ဂိုလ်များကို လူပုဂ္ဂိုလ် သို့မဟုတ် မိသားစုဟု အမည်ပေးထားသည်။",
                         'LOC': "တည်နေရာ အဖွဲ့အစည်းများသည် မြို့များ၊ ပြည်နယ်များ၊ နိုင်ငံများ၊ နိုင်ငံတကာ ဒေသများ၊ ရေပြင်များ၊ တောင်များ စသည်တို့ကဲ့သို့ နိုင်ငံရေးအရ သို့မဟုတ် ပထဝီဝင်အရ သတ်မှတ်ထားသော တည်နေရာများ၏ အမည်များ ဖြစ်ပါသည်။",
                         }
    elif lang == "pt":
        label_details = {'ORG': "As entidades da organização estão limitadas a entidades corporativas, governamentais ou outras entidades organizacionais nomeadas.",
                         'PER': "As entidades de pessoa são pessoas nomeadas ou família.",
                         'LOC': "As entidades de localização são o nome de locais definidos política ou geograficamente, como cidades, províncias, países, regiões internacionais, corpos d'água, montanhas, etc.",
                         }
    elif lang == "ru":
        label_details = {'ORG': "Организационные сущности ограничены названными корпоративными, государственными или другими организационными единицами.",
                         'PER': "Сущности-лица называются лицами или семьями.",
                         'LOC': "Объекты местоположения — это названия политически или географически определенных местоположений, таких как города, провинции, страны, международные регионы, водоемы, горы и т. д.",
                         }
    elif lang == "sw":
        label_details = {'ORG': "Huluki za shirika zinapatikana tu kwa mashirika yaliyotajwa, ya serikali au mashirika mengine.",
                         'PER': "Vyombo vya watu huitwa watu au familia.",
                         'LOC': "Huluki za eneo ni jina la maeneo yaliyoainishwa kisiasa au kijiografia kama vile miji, mikoa, nchi, maeneo ya kimataifa, maeneo ya maji, milima, n.k.",
                         }
    elif lang == "ta":
        label_details = {'ORG': "நிறுவன நிறுவனங்கள் பெயரிடப்பட்ட கார்ப்பரேட், அரசு அல்லது பிற நிறுவன நிறுவனங்களுக்கு மட்டுமே.",
                         'PER': "தனிநபர் நிறுவனங்கள் நபர்கள் அல்லது குடும்பம் என்று பெயரிடப்படுகின்றன.",
                         'LOC': "நகரங்கள், மாகாணங்கள், நாடுகள், சர்வதேச பகுதிகள், நீர்நிலைகள், மலைகள் போன்ற அரசியல் ரீதியாக அல்லது புவியியல் ரீதியாக வரையறுக்கப்பட்ட இடங்களின் பெயர்தான் இருப்பிட நிறுவனங்கள்.",
                         }
    elif lang == "te":
        label_details = {'ORG': "సంస్థ సంస్థలు పేరు పెట్టబడిన కార్పొరేట్, ప్రభుత్వ లేదా ఇతర సంస్థాగత సంస్థలకు పరిమితం చేయబడ్డాయి.",
                         'PER': "వ్యక్తి ఎంటిటీలు వ్యక్తులు లేదా కుటుంబం అని పేరు పెట్టారు.",
                         'LOC': "నగరాలు, ప్రావిన్సులు, దేశాలు, అంతర్జాతీయ ప్రాంతాలు, నీటి వనరులు, పర్వతాలు మొదలైన రాజకీయంగా లేదా భౌగోళికంగా నిర్వచించబడిన స్థానాల పేరు స్థాన ఎంటిటీలు.",
                         }
    elif lang == "th":
        label_details = {'ORG': "เอนทิตีองค์กรจำกัดเฉพาะองค์กรที่มีชื่อ หน่วยงานของรัฐ หรือองค์กรอื่นๆ",
                         'PER': "นิติบุคคลชื่อบุคคลหรือครอบครัว",
                         'LOC': "หน่วยงานที่ตั้งคือชื่อของสถานที่ที่กำหนดทางการเมืองหรือทางภูมิศาสตร์ เช่น เมือง จังหวัด ประเทศ ภูมิภาคระหว่างประเทศ แหล่งน้ำ ภูเขา ฯลฯ",
                         }
    elif lang == "tl":
        label_details = {'ORG': "Limitado ang mga entity ng organisasyon sa mga pinangalanang corporate, government, o iba pang entity ng organisasyon.",
                         'PER': "Ang mga entidad ng tao ay pinangalanang mga tao o pamilya.",
                         'LOC': "Ang mga entity ng lokasyon ay ang pangalan ng mga lokasyong tinukoy sa pulitika o heograpiya gaya ng mga lungsod, lalawigan, bansa, internasyonal na rehiyon, anyong tubig, bundok, atbp.",
                         }
    elif lang == "tr":
        label_details = {'ORG': "Kuruluş varlıkları, adlandırılmış kurumsal, resmi veya diğer kurumsal kuruluşlarla sınırlıdır.",
                         'PER': "Kişi varlıkları kişiler veya aile olarak adlandırılır.",
                         'LOC': "Konum varlıkları, şehirler, iller, ülkeler, uluslararası bölgeler, su kütleleri, dağlar vb. gibi siyasi veya coğrafi olarak tanımlanmış konumların adıdır.",
                         }
    elif lang == "ur":
        label_details = {'ORG': "تنظیمی ادارے نام کارپوریٹ، سرکاری، یا دیگر تنظیمی اداروں تک محدود ہیں۔",
                         'PER': "شخصی اداروں کا نام افراد یا خاندان ہوتا ہے۔",
                         'LOC': "مقام کے ادارے سیاسی یا جغرافیائی طور پر متعین مقامات کا نام ہیں جیسے شہر، صوبے، ممالک، بین الاقوامی علاقے، آبی ذخائر، پہاڑ وغیرہ۔",
                         }
    elif lang == "yo":
        label_details = {'ORG': "Awọn ile-iṣẹ ti ile-iṣẹ ni opin si ajọ-ajọpọ, ti ijọba, tabi awọn ile-iṣẹ eto miiran.",
                         'PER': "Awọn ile-iṣẹ ti eniyan ni orukọ eniyan tabi idile.",
                         'LOC': "Awọn ile-iṣẹ ipo jẹ orukọ awọn ipo iṣelu tabi ti agbegbe gẹgẹbi awọn ilu, awọn agbegbe, awọn orilẹ-ede, awọn agbegbe agbaye, awọn ara omi, awọn oke-nla, ati bẹbẹ lọ.",
                         }
    else:
        raise NotImplementedError
    mrc_f = []
    all_label_dict = []
    for ic, one in enumerate(conll_f):
        context = one[0]
        label_list = one[1]
        label_dict = {x:[] for x in label_details}
        last_label = None
        for il, label_one in enumerate(label_list):
            if last_label is not None and last_label.startswith("I") and not label_one.startswith("I"):
                assert label_id_start == label_id_end
                assert label_start < label_end
                label_dict[label_id_start].append([label_start, label_end])
            elif last_label is not None and last_label.startswith("B") and not label_one.startswith("I"):
                label_dict[label_id_start].append([label_start, label_start])
            if label_one.startswith("B"):
                count += 1
                label_id_start = label_one.split(".")[0].split("-")[1]
                label_start = il
                if il == len(label_list) - 1:
                    label_dict[label_id_start].append([il, il])
            elif label_one.startswith("I"):
                label_id_end = label_one.split(".")[0].split("-")[1]
                label_end = il
                if il == len(label_list) - 1:
                    assert label_id_start == label_id_end
                    assert label_start < label_end
                    label_dict[label_id_start].append([label_start, label_end])


            last_label = label_one
        all_label_dict.append(label_dict)
        for il, label_id in enumerate(label_details):
            entity_label = label_id
            query = label_details[label_id]
            qas_id = '{}.{}'.format(ic, il)
            label_item = label_dict[label_id]
            start_position = [x[0] for x in label_item]
            end_position = [x[1] for x in label_item]
            span_position = ['{};{}'.format(x[0], x[1]) for x in label_item]
            if span_position != []:
                impossible = False
            else:
                impossible = True
            mrc_one = {
                "context": " ".join(context),
                "end_position": end_position,
                "entity_label": entity_label,
                "impossible": impossible,
                "qas_id": qas_id,
                "query": query,
                "span_position": span_position,
                "start_position": start_position,
            }
            mrc_f.append(mrc_one)
    return mrc_f


if __name__ == "__main__":
    for dataset in ['test']:
        for lang in ['af', 'ar', 'bg', 'bn', 'de','el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'he', 'hi', 'hu', 'id', 'it', 'ja', 'jv', 'ka','kk','ko','ml','mr','ms','my','nl','pt','ru','sw','ta','te','th','tl','tr','ur','vi','yo','zh']:
            data_file_path = os.path.join("./panx", dataset + "-" + lang + ".tsv")
            conll_f = read_conll(data_file_path, delimiter="\t")
            mrc_f = conll2mrc(conll_f, lang=lang)
            save_file = os.path.join("./panx", lang + "-mrc-ner" + "." + dataset)
            with open(save_file, 'w') as writer:
                json.dump(mrc_f, writer, ensure_ascii=False, sort_keys=True, indent=2)
    for dataset in ['train', 'dev']:
        for lang in ['en']:
            data_file_path = os.path.join("./panx", dataset + "-" + lang + ".tsv")
            conll_f = read_conll(data_file_path, delimiter="\t")
            mrc_f = conll2mrc(conll_f, lang=lang)
            save_file = os.path.join("./panx", lang + "-mrc-ner" + "." + dataset)
            with open(save_file, 'w') as writer:
                json.dump(mrc_f, writer, ensure_ascii=False, sort_keys=True, indent=2)
# maxlen = 0
# for one in mrc_f:
#         maxlen += len(one['start_position'])
# label_dict = {}
# for one in dataset_item_lst:
#     for label in one[1]:
#         if label.startswith("B"):
#             label_name = label.replace("B-", "")
#             if label_name not in label_dict:
#                 label_dict[label_name] = 0
#             label_dict[label_name] += 1