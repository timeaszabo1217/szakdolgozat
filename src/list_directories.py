import os


def list_directories(base_dir):
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(base_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f'{subindent}{f}')


if __name__ == "__main__":
    base_dir = os.path.abspath('data')
    print(f'Listing contents of: {base_dir}')
    list_directories(base_dir)
