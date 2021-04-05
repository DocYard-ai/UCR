# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Modifications copyright (c) 2021 DocYard Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import logging
import string

import numpy as np

log = logging.getLogger(__name__)


class ClsLabelEncode(object):
    def __init__(self, label_list, **kwargs):
        self.label_list = label_list

    def __call__(self, data):
        label = data["label"]
        if label not in self.label_list:
            return None
        label = self.label_list.index(label)
        data["label"] = label
        return data


class DetLabelEncode(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        import json

        label = data["label"]
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]["points"]
            txt = label[bno]["transcription"]
            boxes.append(box)
            txts.append(txt)
            if txt in ["*", "###"]:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool)

        data["polys"] = boxes
        data["texts"] = txts
        data["ignore_tags"] = txt_tags
        return data

    def order_points_clockwise(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes


class BaseRecLabelEncode(object):
    """ Convert between text-label and text-index """

    def __init__(
        self,
        max_text_length,
        char_dict_location=None,
        lang="ch_sim",
        use_space_char=False,
    ):
        support_lang = [
            "ch_sim",
            "en",
            "en_case",
            "en_number",
            "fr",
            "de",
            "ja",
            "ko",
            "it",
            "es",
            "pt",
            "ru",
            "ar",
            "hi",
            "ug",
            "fa",
            "ur",
            "rs_latin",
            "oc",
            "rs_cyrillic",
            "bg",
            "uk",
            "be",
            "te",
            "kn",
            "ch_tra",
            "ta",
            "mr",
            "ne",
        ]
        assert (
            lang in support_lang
        ), "Only {} are supported now but get {}".format(support_lang, lang)

        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        if lang == "en_case":
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif lang == "en_number":
            # same with ASTER setting (use 94 char).
            self.character_str = string.printable[:62]
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)
        elif lang in support_lang:
            self.character_str = ""
            assert (
                char_dict_location is not None
            ), "char_dict_location should not be None when lang is {}".format(
                lang
            )
            with open(char_dict_location, "rb") as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode("utf-8").strip("\n").strip("\r\n")
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)
        self.lang = lang
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        if len(text) == 0:
            return None
        if self.lang == "en_case":
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                log.warning("{} is not in dict".format(char))
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(
        self,
        max_text_length,
        char_dict_location=None,
        lang="ch",
        use_space_char=False,
        **kwargs
    ):
        super(CTCLabelEncode, self).__init__(
            max_text_length, char_dict_location, lang, use_space_char
        )

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        data["length"] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data["label"] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class AttnLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(
        self,
        max_text_length,
        char_dict_location=None,
        lang="ch",
        use_space_char=False,
        **kwargs
    ):
        super(AttnLabelEncode, self).__init__(
            max_text_length, char_dict_location, lang, use_space_char
        )

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        if text is None:
            return None
        if len(text) >= self.max_text_len:
            return None
        data["length"] = np.array(len(text))
        text = (
            [0]
            + text
            + [len(self.character) - 1]
            + [0] * (self.max_text_len - len(text) - 2)
        )
        data["label"] = np.array(text)
        return data

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, (
                "Unsupport type %s in get_beg_end_flag_idx" % beg_or_end
            )
        return idx


class SRNLabelEncode(BaseRecLabelEncode):
    """ Convert between text-label and text-index """

    def __init__(
        self,
        max_text_length=25,
        char_dict_location=None,
        lang="en_case",
        use_space_char=False,
        **kwargs
    ):
        super(SRNLabelEncode, self).__init__(
            max_text_length, char_dict_location, lang, use_space_char
        )

    def add_special_char(self, dict_character):
        dict_character = dict_character + [self.beg_str, self.end_str]
        return dict_character

    def __call__(self, data):
        text = data["label"]
        text = self.encode(text)
        char_num = len(self.character)
        if text is None:
            return None
        if len(text) > self.max_text_len:
            return None
        data["length"] = np.array(len(text))
        text = text + [char_num - 1] * (self.max_text_len - len(text))
        data["label"] = np.array(text)
        return data

    def get_ignored_tokens(self):
        beg_idx = self.get_beg_end_flag_idx("beg")
        end_idx = self.get_beg_end_flag_idx("end")
        return [beg_idx, end_idx]

    def get_beg_end_flag_idx(self, beg_or_end):
        if beg_or_end == "beg":
            idx = np.array(self.dict[self.beg_str])
        elif beg_or_end == "end":
            idx = np.array(self.dict[self.end_str])
        else:
            assert False, (
                "Unsupport type %s in get_beg_end_flag_idx" % beg_or_end
            )
        return idx
