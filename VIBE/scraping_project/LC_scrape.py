import pandas as pd
import requests

# Parameters
store_ids_list = ['9819','12107','1968','1977','1969','1965','8374']
store_detail_dict={
'1320 s. main st.roswell, NM 88203':'9819',
'1701 n main stroswell, NM 88201':'12107',
'2200 n. main stclovis, NM 88101':'1968',
'1700 w main st suite 1artesia, NM 88210':'1977',
'810 n canalcarlsbad, NM 88220':'1969',
'2400 n grimes ste b12hobbs, NM 88240':'1965',
'616 s. white sands blvd.alamogordo, NM 88310':'8374',
}
menu_version = '1'
base_url = 'https://api.cloud.littlecaesars.com/bff/api/v4/stores'

# Initialize data storage
item_name = []
item_price = []
category = []
store_names=[]
store_ids = []


# Loop through store IDs
for store_name, store_id in store_detail_dict.items():
    url = f"{base_url}/{store_id}/{menu_version}/menu"
    response = requests.get(url)

    if response.status_code == 200:
        print(f"Menu data retrieved successfully for store {store_id}")
        menu_data = response.json()

        categories_data = menu_data.get('menuTypes', [])

        for categoryi in categories_data:
            cat = categoryi.get('typeDescription', 'Unknown')

            menu_items = categoryi.get('menuItems', [])
            for item in menu_items:
                name = item.get('itemName', 'Unknown')
                price = item.get('price', 0)

                item_name.append(name)
                item_price.append(price)
                store_names.append(store_name)
                store_ids.append(store_id)
                category.append(cat)

    else:
        print(f"Request failed for store {store_id} with status code {response.status_code}")
        print(response.text)

# Create DataFrame
df = pd.DataFrame({
    'item_name': item_name,
    'price': item_price,
    'category': category,
    'store_id': store_ids,
    'store_name':store_names
})

# Save to CSV
df.to_csv('lc_scrape.csv', index=False)
print("Saved data to lc_scrape.csv")
 