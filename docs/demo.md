---
hide:
  - navigation # Hide navigation
  - toc        # Hide table of contents
---

Every Subsection here, contains a specific setting in which Experiments were conducted and corresponding Results reported. All the results are Reproducible with their respective Jupyter Notebook/Colab link attached to it! 

## Default Configuration [:fontawesome-brands-github:](coming_soon.md)

=== "Document 1"
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR()
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output'` 
    ![Placeholder](static/images/demo_results/ocr_laura-lefurgey-smith-r5NG8QBkOhQ-unsplash.jpg.jpg){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

=== "Scene Text 1"

    ![Placeholder](static/images/demo_results/ocr_russian.png.png){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR()
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

=== "Document 2"

    ![Placeholder](static/images/demo_results/ocr_russian.png.png){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR()
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

=== "Scene Text 2"

    ![Placeholder](static/images/demo_results/ocr_russian.png.png){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR()
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

## Whitelist/Blacklist characters [:fontawesome-brands-github:](coming_soon.md)

=== "Whitelist 'en' Characters"

    ![Placeholder](static/images/demo_results/ocr_laura-lefurgey-smith-r5NG8QBkOhQ-unsplash.jpg.jpg){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR(rec_whitelist='abcdefghij')
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output' --rec_whitelist='abcdefghij'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

=== "Blacklist Numbers"

    ![Placeholder](static/images/demo_results/ocr_russian.png.png){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR(rec_blacklist='0123456789')
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output' --rec_blacklist '0123456789'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

## Mobile/Server Configuration [:fontawesome-brands-github:](coming_soon.md)

=== "Mobile Backend"

    ![Placeholder](static/images/demo_results/ocr_russian.png.png){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR(rec_blacklist='0123456789')
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output' --rec_blacklist '0123456789'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

=== "Server Backend"

    ![Placeholder](static/images/demo_results/ocr_russian.png.png){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR(rec_blacklist='0123456789')
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output' --rec_blacklist '0123456789'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

## Multiple Language Support [:fontawesome-brands-github:](coming_soon.md)

=== "English 'en'"

    ![Placeholder](static/images/demo_results/ocr_laura-lefurgey-smith-r5NG8QBkOhQ-unsplash.jpg.jpg){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR()
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

=== "German 'de'"

    ![Placeholder](static/images/demo_results/ocr_russian.png.png){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR()
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

=== "French 'fr'"

    ![Placeholder](static/images/demo_results/ocr_russian.png.png){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR()
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

=== "Hindi, 'hi'"

    ![Placeholder](static/images/demo_results/ocr_russian.png.png){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR()
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

=== "Russian, 'ru'"

    ![Placeholder](static/images/demo_results/ocr_russian.png.png){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR()
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

=== "Korean, 'ko'"

    ![Placeholder](static/images/demo_results/ocr_russian.png.png){align="left"; loading="lazy"}
    !!! info inline end ""
        language....<span style="color:#FF8856; font-style: italic">=="ch_sim"==</span>.... | type....<span style="color:#FF8856; font-style: italic">=="server"==</span>.... | det....<span style="color:#FF8856; font-style: italic">=="CRAFT"==</span>.... | rec....<span style="color:#FF8856; font-style: italic">=="CRNN"==</span>.... | cls....<span style="color:#FF8856; font-style: italic">=="CLS"==</span>.... |
    !!! success inline end "" 
        Python:
        ``` python linenums="1"
        from ucr import UCR
        ocr = UCR()
        result = ocr(i='input', o='output')
        ```
        CLI: `ucr -i 'input' -o 'output'` 
    !!! tip inline end "Prediction Stats"
        cpu_inference....<span style="color:#FF8856; font-style: italic">5s</span>.... | gpu_inference....<span style="color:#FF8856; font-style: italic">1s</span>.... | CER....<span style="color:#FF8856; font-style: italic">1.2</span>.... | WER....<span style="color:#FF8856; font-style: italic">1.1</span>.... 

