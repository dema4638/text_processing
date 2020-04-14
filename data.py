import nltk
from nltk.tokenize import sent_tokenize
from urllib import request
from bs4 import BeautifulSoup
import csv
import re


def writeToFile(author,doc, id):
    i=0
    file = open('train1.csv', 'a',newline='',encoding='utf-8')
    writer = csv.writer(file,quoting=csv.QUOTE_ALL)
    for i, token in enumerate(doc):
        writer.writerow([i+id,token,author])
    file.close()
    return i+id+1;
        
def getSentences(url, beginning, ending):
    response = request.urlopen(url)
    text1 = response.read().decode('utf8')
    start = text1.find(beginning)
    end = text1.find(ending)
    text1 = text1[start:end]
    text = BeautifulSoup(text1, 'html').get_text()
    text=re.sub(r'\s+', ' ', text)
    sentences = sent_tokenize(text)
    return sentences

id = 0;
file = open('train1.csv', 'a',newline='',encoding='utf-8')
writer = csv.writer(file,quoting=csv.QUOTE_ALL)
writer.writerow(["ID","Text","Author"])
file.close()
#Jonas Biliunas: Kliudziau
url = "http://antologija.lt/text/jonas-biliunas-liudna-pasaka/13"
sentences = getSentences(url, "Tat buvo nedidelė", "1905.V.22");
id = writeToFile("Biliunas",sentences,id)
#Jonas Biliunas: Laimes ziburys
url = "http://antologija.lt/text/jonas-biliunas-liudna-pasaka/12"
sentences = getSentences(url,"Ant aukšto stataus kalno","Ciurichas, 1905.IV.15");
id = writeToFile("Biliunas",sentences,id)
#Jonas Biliunas: Ubagas
url = "http://antologija.lt/text/jonas-biliunas-liudna-pasaka/18"
sentences = getSentences(url,"Parvažiavęs iš užusienių Lietuvon","Niūronys, 1906.VI.8");
id = writeToFile("Biliunas",sentences,id)
#Jonas Biliunas: Nemunu
url = "http://antologija.lt/text/jonas-biliunas-liudna-pasaka/09"
sentences = getSentences(url,"Vienodai ir nuobodžiai čiuksėdama","Ciurichas, 1905.IV.2");
id = writeToFile("Biliunas",sentences,id)
#Jonas Biliunas: Lazda
url = "http://antologija.lt/text/jonas-biliunas-liudna-pasaka/17"
sentences = getSentences(url,"Dabar gimtasai mano sodžius","Guli jinai ant lentynos klėty, ir niekas jos neliečia.");
id = writeToFile("Biliunas",sentences,id)
#Kristijonas Donelaitis: Metai - Pavasario linksmybes
url = "http://antologija.lt/text/kristijonas-donelaitis-metai"
sentences = getSentences(url, "Jau saulelė vėl atkopdama", "Gana");
id = writeToFile("Donelaitis",sentences,id)
#Kristijonas Donelaitis: Metai - Vasaros darbai
url = "http://antologija.lt/text/kristijonas-donelaitis-metai/2"
sentences = getSentences(url, ",,Sveiks, svieteli margs!", "Gana");
#Kristijonas Donelaitis: Metai - Rudenio gėrybės
url = "http://antologija.lt/text/kristijonas-donelaitis-metai/3"
sentences = getSentences(url, "Ant saulelė vėl nuo", "Grečną zopostėlį mums dar pasilikusį matom?");
id = writeToFile("Donelaitis",sentences,id)
#Zemaite: Laimė nutekėjimo - Neturėjo geros motynos
url = "http://antologija.lt/text/zemaite-laime-nutekejimo/1"
sentences = getSentences(url, "Per keletą varstų nuo sodos", "1895 m.");
id = writeToFile("Zemaite",sentences,id)
#Zemaite: Laimė nutekėjimo - Tofylis
url = "http://antologija.lt/text/zemaite-laime-nutekejimo/10"
sentences = getSentences(url, "Dabar Zosė jau linksma, užganėdyta", "1897 m.");
id = writeToFile("Zemaite",sentences,id)
#Zemaite: Laimė nutekėjimo - Sutkai
url = "http://antologija.lt/text/zemaite-laime-nutekejimo/11"
sentences = getSentences(url, "Pavasarelyj gražus ir tykus oras", "Eikim, sutems mumis.");
id = writeToFile("Zemaite",sentences,id)
#Zemaite: Laimė nutekėjimo - Sučiuptas velnias
url = "http://antologija.lt/text/zemaite-laime-nutekejimo/14"
sentences = getSentences(url, "Kad speigas, tai speigas!", "1898 m.");
id = writeToFile("Zemaite",sentences,id)
#Zemaite: Laimė nutekėjimo - Marti
url = "http://antologija.lt/text/zemaite-laime-nutekejimo/2"
sentences = getSentences(url, "Besitaisant, besibrūzdant", "Man mat nepakelti!..");
id = writeToFile("Zemaite",sentences,id)
