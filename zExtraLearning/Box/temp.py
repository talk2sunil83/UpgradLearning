# %%
# curl https://api.box.com/2.0/folders/0 -H \
# "Authorization: Bearer DO4LtcgZLVAFohpSyGw9aExPsuoIWkU6"

from boxsdk import OAuth2, Client
import os
# %%

CLIENT_ID = "sv5ceytsvpf5r56rmcc599yvpkvuza3r"
CLIENT_SECRET = "qOZYkYfTzUZViOQ9VGUATfAUMFGBGfuP"
DEVELOPER_TOKEN = "PcX5JgEityWYvLuZ7UEaLZbg0f1T0sG1"
folder_id = "124287328238"
# May be auth method need to change DEVELOPER_TOKEN changes periodically
auth = OAuth2(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    access_token=DEVELOPER_TOKEN,
)
client = Client(auth)

user = client.user().get()
print('The current user ID is {0}'.format(user.id))
# %%

# Ref:  https://github.com/box/box-python-sdk/blob/master/demo/example.py


indentation_base = 4


def run_folder_examples(client, folder_id='0', level=0, base_path=''):
    indentation = " " * (level*indentation_base)
    folder = client.folder(folder_id).get()  # Get root folder
    folder_name = folder.name
    target_folder = base_path+"/"+folder_name if base_path != '' else folder_name
    os.makedirs(target_folder, exist_ok=True)
    # items = test_folder.get_items(limit=100, offset=0) # items could be paged
    items = folder.get_items()
    # print('This is the first 100 items in the root folder:')
    print(f'{indentation}Items in the folder:')
    files_in_folder = []
    folders_in_folder = []
    for item in items:
        item_id = item.id
        level += 1
        item_type_name = item.type.capitalize()
        print(f"{' ' * (level*indentation_base)}Id:{item_id}, Name:{item.name}, Type:{item_type_name}")
        if item_type_name == "Folder":
            folders_in_folder.append((item_id, level, target_folder))
            # run_folder_examples(client, item_id, level)
        elif item_type_name == "File":
            files_in_folder.append(item)

    for item in files_in_folder:
        # Ref: https://github.com/box/box-python-sdk/blob/master/docs/usage/files.md#download-a-file
        with open(target_folder+"/"+item.name, "wb") as f:
            file_content = client.file(item.id).content()  # file could be get in chunks
            f.write(file_content)
    for folder_id, level, base_path in folders_in_folder:
        run_folder_examples(client, folder_id, level, base_path)


run_folder_examples(client)

# %%
