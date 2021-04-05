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

import string

import numpy as np
import torch


class BaseRecLabelDecode(object):
    """ Convert between text-label and text-index """

    def __init__(
        self,
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
            "EN",
        ]
        assert (
            lang in support_lang
        ), "Only {} are supported now but get {}".format(support_lang, lang)

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

        else:
            raise NotImplementedError
        self.lang = lang
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1]
                        == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(
                    self.character[int(text_index[batch_idx][idx])]
                )
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def get_ignored_tokens(self):
        return [0]  # for ctc blank


class CTCLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(
        self,
        char_dict_location=None,
        lang="ch_sim",
        use_space_char=False,
        **kwargs
    ):
        super(CTCLabelDecode, self).__init__(
            char_dict_location, lang, use_space_char
        )

    def __call__(self, preds, label=None, *args, **kwargs):
        if torch.is_tensor(preds):
            preds = preds.numpy()
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        if label is None:
            return text
        label = self.decode(label)
        return text, label

    def add_special_char(self, dict_character):
        dict_character = ["blank"] + dict_character
        return dict_character


class AttnLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(
        self,
        char_dict_location=None,
        lang="ch",
        use_space_char=False,
        **kwargs
    ):
        super(AttnLabelDecode, self).__init__(
            char_dict_location, lang, use_space_char
        )

    def add_special_char(self, dict_character):
        self.beg_str = "sos"
        self.end_str = "eos"
        dict_character = dict_character
        dict_character = [self.beg_str] + dict_character + [self.end_str]
        return dict_character

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        [beg_idx, end_idx] = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if int(text_index[batch_idx][idx]) == int(end_idx):
                    break
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1]
                        == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(
                    self.character[int(text_index[batch_idx][idx])]
                )
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)
            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def __call__(self, preds, label=None, *args, **kwargs):
        """
        text = self.decode(text)
        if label is None:
            return text
        else:
            label = self.decode(label, is_remove_duplicate=False)
            return text, label
        """
        if torch.is_tensor(preds):
            preds = preds.numpy()

        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
        text = self.decode(preds_idx, preds_prob, is_remove_duplicate=False)
        if label is None:
            return text
        label = self.decode(label, is_remove_duplicate=False)
        return text, label

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
                "unsupport type %s in get_beg_end_flag_idx" % beg_or_end
            )
        return idx


class SRNLabelDecode(BaseRecLabelDecode):
    """ Convert between text-label and text-index """

    def __init__(
        self,
        char_dict_location=None,
        lang="en_case",
        use_space_char=False,
        **kwargs
    ):
        super(SRNLabelDecode, self).__init__(
            char_dict_location, lang, use_space_char
        )

    def __call__(self, preds, label=None, *args, **kwargs):
        pred = preds["predict"]
        char_num = len(self.character_str) + 2
        if torch.is_tensor(pred):
            pred = pred.numpy()
        pred = np.reshape(pred, [-1, char_num])

        preds_idx = np.argmax(pred, axis=1)
        preds_prob = np.max(pred, axis=1)

        preds_idx = np.reshape(preds_idx, [-1, 25])

        preds_prob = np.reshape(preds_prob, [-1, 25])

        text = self.decode(preds_idx, preds_prob)

        if label is None:
            text = self.decode(
                preds_idx, preds_prob, is_remove_duplicate=False
            )
            return text
        label = self.decode(label)
        return text, label

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)

        for batch_idx in range(batch_size):
            char_list = []
            conf_list = []
            for idx in range(len(text_index[batch_idx])):
                if text_index[batch_idx][idx] in ignored_tokens:
                    continue
                if is_remove_duplicate:
                    # only for predict
                    if (
                        idx > 0
                        and text_index[batch_idx][idx - 1]
                        == text_index[batch_idx][idx]
                    ):
                        continue
                char_list.append(
                    self.character[int(text_index[batch_idx][idx])]
                )
                if text_prob is not None:
                    conf_list.append(text_prob[batch_idx][idx])
                else:
                    conf_list.append(1)

            text = "".join(char_list)
            result_list.append((text, np.mean(conf_list)))
        return result_list

    def add_special_char(self, dict_character):
        dict_character = dict_character + [self.beg_str, self.end_str]
        return dict_character

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
                "unsupport type %s in get_beg_end_flag_idx" % beg_or_end
            )
        return idx
