import requests
from bs4 import BeautifulSoup
import os

species_name_1 = input('Enter the name of the Species: ')
species_name = species_name_1.split()
name = ''

for i in range(len(species_name)):
    name += species_name[i] + '+'

name = name[:-1]

last_page = input('Enter the last page number: ')

main_url = f'https://www.xeno-canto.org/species/{name}?pg='
download_url = 'https://www.xeno-canto.org'
download_dir = "Birdcalls/"
download_dir += species_name_1 + '/'

if not os.path.exists(download_dir):
    os.mkdir(download_dir)
    print('New directory created.')
else:
    print('Directory exists.')

count_main = 1

for j in range(1, int(last_page) + 1):
    count_sub = 1
    url = main_url + str(j)
    print(url)

    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    hyper = soup.findAll('a')

    for i in range(0, len(soup.findAll('a'))):

        try:
            one_a_tag = soup.findAll('a')[i]
            link = one_a_tag['href']

            if link.split('/')[-1] == 'download':
                full_link = download_url + link
                filename = download_dir + full_link.split('/')[-2] + '.mp3'
                r = requests.get(full_link, stream=True)

                with open(filename, 'wb') as mp3:
                    for chunk in r.iter_content(chunk_size=1024 * 2):
                        if chunk:
                            mp3.write(chunk)

                print(count_main, count_sub, full_link)

                count_sub += 1
                count_main += 1

        except Exception as e:
            print(e)

    print(f'Done Parsing Page {j}.')
print(f'Finished Parsing {species_name_1}.')
print(f'Got {count_main - 1} data files.')
