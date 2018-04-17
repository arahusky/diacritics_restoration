Scripts for creating training corpora for various languages
----

This folder stores scripts that create corpus for training diacritics restoration systems. The data are downloaded from two web sources.

To prepare data from both sources for selected language (see table below), run:

```
bash prepare_data_for_language.sh slk sk my_data_dir

```

To prepare Wiki data only, run:

```
bash process_w2c.sh slk sk my_data_dir/sk
```


To create additional training set from Web, run:
-----

```
bash process_statmt.sh et.2015_27 et /home/arahusky/troja/w2c_data/et n
```


```
W2C languages:
Afrikaans			afr
Tosk Albanian			als
Amharic			amh
Arabic			ara
Aragonese			arg
Egyptian Arabic			arz
Asturian			ast
Azerbaijani			aze
Belarusian			bel
Bengali			ben
Bosnian			bos
Bishnupriya			bpy
Breton			bre
Buginese			bug
Bulgarian			bul
Catalan			cat
Cebuano			ceb
Czech			ces
Chuvash			chv
Corsican			cos
Welsh			cym
Danish			dan
German			deu
Dimli (individual language)			diq
Modern Greek (1453-)			ell
English			eng
Esperanto			epo
Estonian			est
Basque			eus
Faroese			fao
Persian			fas
Finnish			fin
French			fra
Western Frisian			fry
Gan Chinese			gan
Scottish Gaelic			gla
Irish			gle
Galician			glg
Gilaki			glk
Gujarati			guj
Haitian			hat
Serbo-Croatian			hbs
Hebrew			heb
Fiji Hindi			hif
Hindi			hin
Croatian			hrv
Upper Sorbian			hsb
Hungarian			hun
Armenian			hye
Ido			ido
Interlingua (International Auxiliary Language Association)			ina
Indonesian			ind
Icelandic			isl
Italian			ita
Javanese			jav
Japanese			jpn
Kannada			kan
Georgian			kat
Kazakh			kaz
Korean			kor
Kurdish			kur
Latin			lat
Latvian			lav
Limburgan			lim
Lithuanian			lit
Lombard			lmo
Luxembourgish			ltz
Malayalam			mal
Marathi			mar
Macedonian			mkd
Malagasy			mlg
Mongolian			mon
Maori			mri
Malay (macrolanguage)			msa
Burmese			mya
Neapolitan			nap
Low German			nds
Nepali			nep
Newari			new
Dutch			nld
Norwegian Nynorsk			nno
Norwegian			nor
Occitan (post 1500)			oci
Ossetian			oss
Pampanga			pam
Piemontese			pms
Polish			pol
Portuguese			por
Quechua			que
Romanian			ron
Russian			rus
Yakut			sah
Sicilian			scn
Scots			sco
Slovak			slk
Slovenian			slv
Spanish			spa
Albanian			sqi
Serbian			srp
Sundanese			sun
Swahili (macrolanguage)			swa
Swedish			swe
Tamil			tam
Tatar			tat
Telugu			tel
Tajik			tgk
Tagalog			tgl
Thai			tha
Turkish			tur
Ukrainian			ukr
Urdu			urd
Uzbek			uzb
Venetian			vec
Vietnamese			vie
Volapük			vol
Waray (Philippines)			war
Walloon			wln
Yiddish			yid
Yoruba			yor
Chinese			zho
```
