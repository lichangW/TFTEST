#!/Users/cj/workspace/workspace/bin/python
# -*- coding:utf-8 -*-
import urllib
import urllib2
from bs4 import BeautifulSoup
import logging,sys,os
import requests
import base64
import time

reload(sys)
sys.setdefaultencoding('utf8')
logging.basicConfig(stream=sys.stdout,level=logging.DEBUG)

osite="http://www.zimuku.cn"
subs=["/newsubs?t=tv&ad=1","/newsubs?t=mv&ad=1"]
data="/Users/cj/workspace/TFTEST/seq2seq/chinese2english/mv_scripts/"

def get_soup(path):

    try:
        req = urllib2.Request(path)
        resp = urllib2.urlopen(req)
        page = resp.read()
        soup = BeautifulSoup(page)
    except Exception as _e:
        logging.error("get_soup failed, path:%s, error:%s",path,str(_e))
        return  None
    return  soup

def single_mv(single_mv_hrefs):
    for single_mv_td in single_mv_hrefs:
        single_mv_href=single_mv_td.find("a",{"class":"tooltips"})
        if single_mv_href is None:
            logging.error("single_mv_href is None")
            continue
        single_mv_subs = osite + single_mv_href.attrs["href"]
        logging.info("single_mv_subs new path:%s", single_mv_subs)
        single_mv_subs_soup = get_soup(single_mv_subs)
        if single_mv_subs_soup is None:
            logging.error("single_mv_subs_soup is None")
            continue
        single_mv_scripts=single_mv_subs_soup.find_all("tr")
        if single_mv_scripts is None:
            logging.error("single_mv_scripts is None")
            continue
        for signle_item in single_mv_scripts:
            signle_item_href=signle_item.find("td",{"class":"first"})
            if signle_item_href is None:
                logging.error("signle_item_href is None")
                continue
            ahref=signle_item_href.find("a")
            if ahref is None:
                logging.error("ahref is None")
                continue

            detail_href=ahref.attrs["href"]
            if detail_href == "":
                continue
            detail_href=osite+detail_href
            logging.info("download page:%s",detail_href)

            languages=signle_item.find("td",{"class":"tac lang"})
            if languages is None:
                continue

            english=False
            chinese=False
            for img in languages.children:
                logging.info("alt: %s",img.attrs["alt"].strip().decode("utf-8"))
                if img.attrs["alt"].strip().decode("utf-8")==u"双语字幕":
                    mv_details(detail_href)
                if img.attrs["alt"].strip().decode("utf-8")==u"简体中文字幕" or img.attrs["alt"].strip().decode("utf-8")==u"繁體中文字幕":
                    chinese=True
                if img.attrs["alt"].strip().decode("utf-8")==u"English字幕":
                    english=True
                if english and chinese:
                    mv_details(detail_href)

def mv_details(href):
    logging.info("in mv_details: %s",href)
    detail = get_soup(href)
    if detail is None:
        logging.info("get empty detail page: %s",href)
        return
    urls=[]
    url1=detail.find("a",{"id":"down1"})
    if url1 is not None:
        urls.append(osite+url1["href"])
    url2 = detail.find("a", {"id": "down2"})
    if url2 is not None:
        urls.append(osite+url2["href"])
    if len(urls) != 0:
        download_source(urls)

def download_source(urls,storage=data):

        for l in urls:
            try:
                resp=requests.get(l)
                if resp.status_code/100!=2 or len(resp.content)==0:
                    logging.error("pull scripts error,status code:%d, content length: %d",resp.status_code,len(resp.content))
                    continue

                filename=""
                disposition_splits=resp.headers.get("Content-Disposition","").split("=",2)
                if len(disposition_splits)>=2:
                    filename=disposition_splits[1]
                if filename == "":
                    logging.error("empty file name, hearders {}".format(resp.headers))
                    continue
                # filename=base64.b64encode(str(time.time())+resp.content[:20],altchars="-_")
                filename=data+filename
                file=open(filename,"ab+")
                file.write(resp.content)
                file.close()
                logging.info("get one script:%s",filename)
                break   #break if success
            except Exception as _e:
                logging.error("requests.get exception, url:%s, error:%s",l,str(_e))
                continue

def crawler():

    viewed_page={}
    for sub in subs:
        logging.info("main theme: %s",osite + sub)
        soup=get_soup(osite + sub)
        if soup is None:
            continue
        pages=soup.find_all("a", {"class": "num"})
        for pg in pages:
            single_mv_hrefs=soup.find_all("td",{"class":"first"})
            if single_mv_hrefs!=None:
                single_mv(single_mv_hrefs)
            npage= osite+pg.attrs["href"]
            logging.info("new page under main theme:%s",npage)
            if npage.strip()=="":
                continue
            print "new page",npage
            soup=get_soup(npage)

if __name__=="__main__":
    crawler()