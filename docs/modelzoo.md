---
hide:
  - toc # Hide navigation
---
!!! tip "Tip"
    Pretrained `en_number` model ** does not** support special characters.  
    To use special characters, use any other model (preferrably `ch_sim`) and **whitelist english characters** as shown below:  
    Python:
    ``` python linenums="1"
    import string
    from ucr import UCR
    ocr = UCR(l='ch_sim', type='server', whitelist=string.printable[:-6] # use type='mobile' for small model
    # string.printable[:-6] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~"
    result = ocr.predict('input', o='output')
    ```
    CLI: `ucr predict 'input' -o='output' -l='ch_sim' --type='server' --whitelist=string.printable[:-6]` 
## Models List

|    <span style="font-weight:bold; font-size: 125%">Language</span>                  | <span style="font-weight:bold; font-size: 125%">Lang-id</span>     | <span style="font-weight:bold; font-size: 125%">Dict Link</span>   | <span style="font-weight:bold; font-size: 125%">Model Link</span>
|----------------------|----------------------|----------------------|----------------------|
| Arabic |	`ar` | [ar_dict.txt](ar.md) |[mobile](mobile_link.md) 
| Belarusian |	`be` | [be_dict.txt](be.md) |[mobile](mobile_link.md)
| Bulgarian |	`bg` | [bg_dict.txt](bg.md) |[mobile](mobile_link.md)
| Chinese (Simplified)|`ch_sim`|[ch_sim_dict.txt](en.md)|[mobile](mobile_link.md)/[server](server_link.md)
| Chinese (Traditional) | `ch_tra` | [ch_tra_dict.txt](ch_tra.md) |[mobile](mobile_link.md)
| German |	`de` | [de_dict.txt](de.md) |[mobile](mobile_link.md)
| English (No Symbols) |	`en_number` | [en_number_dict.txt](en_number.md) |[mobile](mobile_link.md)
| English |	`en` | [en_dict.txt](en.md) |[mobile](mobile_link.md)
| Spanish |	`es` | [es_dict.txt](es.md) |[mobile](mobile_link.md)
| Persian (Farsi) | `fa` | [farsi_dict.txt](Farsi.md) |[mobile](mobile_link.md) | 
| French |	`fr` | [fr_dict.txt](fr.md) |[mobile](mobile_link.md)
| Hindi |	`hi` | [hi_dict.txt](hi.md) |[mobile](mobile_link.md)
| Italian |	`it` | [it_dict.txt](it.md) |[mobile](mobile_link.md)
| Japanese |	`ja` | [ja_dict.txt](ja.md) |[mobile](mobile_link.md)
| Kannada |	`kn` | [kn_dict.txt](kn.md) |[mobile](mobile_link.md)
| Korean |	`ko` | [ko_dict.txt](ko.md) |[mobile](mobile_link.md)
| Marathi |	`mr` | [mr_dict.txt](mr.md) |[mobile](mobile_link.md)
| Nepali |	`ne` | [ne_dict.txt](ne.md) |[mobile](mobile_link.md)
| Occitan |	`oc` | [oc_dict.txt](oc.md) |[mobile](mobile_link.md)
| Portuguese |	`pt` | [pt_dict.txt](pt.md) |[mobile](mobile_link.md)
| Russian |	`ru` | [ru_dict.txt](ru.md) |[mobile](mobile_link.md)
| Serbian (cyrillic) | `rs_cyrillic` | [cyrillic_dict.txt](cyrillic.md) | [mobile](mobile_link.md)	
| Serbian (latin) | `rs_latin` | [latin_dict.txt](latin.md) |[mobile](mobile_link.md) | [mobile](mobile_link.md)	
| Tamil |	`ta` | [ta_dict.txt](ta.md) | [mobile](mobile_link.md)
| Telugu |	`te` | [te_dict.txt](te.md) | [mobile](mobile_link.md)
| Uyghur |	`ug` | [ug_dict.txt](ug.md) | [mobile](mobile_link.md)
| Ukranian |	`uk` | [uk_dict.txt](uk.md) | [mobile](mobile_link.md)
| Urdu |	`ur` | [ur_dict.txt](ur.md) | [mobile](mobile_link.md)