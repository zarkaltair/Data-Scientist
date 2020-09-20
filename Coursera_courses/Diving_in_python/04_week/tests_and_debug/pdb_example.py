import re
import requests


def main(site_url,substring):
    # import pdb
    # pdb.set_trace()
    
    site_code = get_site_code(site_url)
    matching_substrings = get_matching_substrings(site_code, substring)
    print(f'{substring} found {len(matching_substrings)} times in {site_url}')

def get_site_code(site_url):
    if not site_url.startswith('http'):
        site_url = 'http://' + site_url
    return requests.get(site_url).text

def get_matching_substrings(source, substring):
    return re.findall(substring, source)


main('mail.ru', 'script')
