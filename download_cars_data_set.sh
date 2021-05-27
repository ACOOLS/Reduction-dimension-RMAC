#!/bin/bash
mkdir cars_cools
cd cars_cools &&  wget https://github.com/ACOOLS/Memoire/releases/download/1/partie1.zip && wget https://github.com/ACOOLS/Memoire/releases/download/1/partie2.zip && wget https://github.com/ACOOLS/Memoire/releases/download/1/partie3.zip && wget https://github.com/ACOOLS/Memoire/releases/download/1/partie4.zip \
&& unzip partie1.zip && unzip partie2.zip && unzip partie3.zip  &&  unzip partie4.zip \
mv partie1/* .  &&  mv partie2/* .  &&  mv partie3/* .  &&  mv partie4/* . \
&&  rm -r partie*
