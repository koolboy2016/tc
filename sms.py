# -*- coding: utf-8 -*-

import httplib
import urllib
import json

sms_host = "sms.yunpian.com"
voice_host = "voice.yunpian.com"
port = 443
version = "v2"
user_get_uri = "/" + version + "/user/get.json"
sms_send_uri = "/" + version + "/sms/single_send.json"
sms_tpl_send_uri = "/" + version + "/sms/tpl_single_send.json"
sms_voice_send_uri = "/" + version + "/voice/send.json"
apikey = "7df4b84ffd47f5d5919ce3cc8f892d88"
mobile = "18868831809"

def send_sms(text):
    text = "【KUE】提醒:"+text;
    params = urllib.urlencode({'apikey': apikey, 'text': text, 'mobile': mobile})
    headers = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain"}
    conn = httplib.HTTPSConnection(sms_host, port=port, timeout=30)
    conn.request("POST", sms_send_uri, params, headers)
    response = conn.getresponse()
    response_str = response.read()
    conn.close()
    return response_str

