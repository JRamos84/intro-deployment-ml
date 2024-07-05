import os
from base64 import b64decode

def main():
    key = os.environ.get('SERVICE_ACCOUNT_KEY')
    if key:
        key_bytes = b64decode(key)
        with open('path.json', 'wb') as json_file:
            json_file.write(key_bytes)
        print(os.path.realpath('path.json'))
    else:
        print("SERVICE_ACCOUNT_KEY environment variable not found")

if __name__ == '__main__':
    main()
