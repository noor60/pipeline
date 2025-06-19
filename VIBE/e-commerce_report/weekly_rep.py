import pandas as pd
# NOTE:  end of week/sunday

data = pd.read_csv(
    "./weekly_sales/transactional data/june_11.csv",
    parse_dates=["BusinessDate"],
    index_col="BusinessDate",
)
data=data[data.index> "2025-05-25 00:00:00"]


# sheet 1
def brand_level_calculations(data):
    brand = (
        data[["NetAmount", "OrderId"]]
        .resample("W")
        .agg({"NetAmount": "sum", "OrderId": "nunique"})
        .reset_index()
    )
    
    brand['BusinessDate'].unique()

    brand["Average Order Value"] = brand["NetAmount"] / brand["OrderId"]
    brand["Average Order Value"] = brand["Average Order Value"].round(2)
    brand["WoW Change"] = brand["NetAmount"].pct_change().multiply(100).round(1)

    brand["Brand"] = "Little Caesars"
    brand["Store Count"] = data.FranchiseStore.nunique()
    brand.rename(
        columns={
            "NetAmount": "Total Revenue",
            "OrderId": "Total Orders",
            "BusinessDate": "Week",
        },
        inplace=True,
    )
    brand[
        [
            "Week",
            "Brand",
            "Store Count",
            "Total Revenue",
            "Total Orders",
            "Average Order Value",
            "WoW Change",
        ]
    ].to_csv("./output_files/weekly_summary_sheet1.csv", index=False)
    print('Weekly summary sheet 1 created successfully!')

brand_level_calculations(data)


# sheet 2
def get_top_channel(store_info,topchannel=True):
    item_group = (
        store_info.groupby(["FranchiseStore", "BusinessDate", "MenuItemName"])[
            "NetAmount"
        ]
        .sum()
        .reset_index()
        .sort_values(
            ["FranchiseStore", "BusinessDate", "NetAmount"],
            ascending=[True, True, False],
        )
    )

    item_group["ItemRank"] = item_group.groupby(["FranchiseStore", "BusinessDate"])[
        "NetAmount"
    ].rank(method="first", ascending=False)
    top2_items = item_group[item_group["ItemRank"] <= 2]

    # Combine into single string
    top2_items_agg = (
        top2_items.groupby(["FranchiseStore", "BusinessDate"])["MenuItemName"]
        .apply(lambda x: ", ".join(x))
        .reset_index(name="Top2MenuItems")
    )
    if topchannel:
        method_group = (
            store_info.groupby(["FranchiseStore", "BusinessDate", "OrderPlacedMethod"])[
                "NetAmount"
            ]
            .sum()
            .reset_index()
            .sort_values(
                ["FranchiseStore", "BusinessDate", "NetAmount"],
                ascending=[True, True, False],
            )
        )

        top_methods = (
            method_group.groupby(["FranchiseStore", "BusinessDate"])
            .first()
            .reset_index()[["FranchiseStore", "BusinessDate", "OrderPlacedMethod"]]
            .rename(columns={"OrderPlacedMethod": "TopOrderMethod"})
        )

        final_result = pd.merge(
            top2_items_agg, top_methods, on=["FranchiseStore", "BusinessDate"]
        )

        final_result = final_result.sort_values(["FranchiseStore", "BusinessDate"])

        print(final_result["TopOrderMethod"].unique())

        final_result.sort_values(by=["FranchiseStore", "BusinessDate"], inplace=True)

        # final_result.rename(columns={'BusinessDate': 'Week'}, inplace=True)
        # store.sort_values(by=['FranchiseStore', 'Week'], inplace=True)
        return final_result
    return top2_items_agg


def store_level_calculations(data):
    store = (
        data[["NetAmount", "OrderId", "FranchiseStore"]]
        .groupby("FranchiseStore")
        .resample("W")
        .agg({"NetAmount": "sum", "OrderId": "nunique"})
        .reset_index()
    )
    store.sort_values(by=["BusinessDate"], inplace=True)

    store["Average Order Value"] = store["NetAmount"] / store["OrderId"]
    store["Average Order Value"] = store["Average Order Value"].round(2)
    
    store['WoW Change'] = (
    store.sort_values(['FranchiseStore','BusinessDate'])  # Ensure proper order
      .groupby(['FranchiseStore'])['NetAmount']
      .pct_change()  # Calculate WoW change within each group
      .multiply(100)
      .round(1)
        )
    store[(store['FranchiseStore']=='01944-00006') ]

    store["Brand"] = "Little Caesars"

    # top channel/ top item and leasrt performing store
    store_info = (
        data[["FranchiseStore", "MenuItemName", "OrderPlacedMethod", "NetAmount"]]
        .groupby(["FranchiseStore", "MenuItemName", "OrderPlacedMethod"])
        .resample("W")
        .agg({"NetAmount": "sum"})
        .reset_index()
    )

    store_info.sort_values(by=["BusinessDate", "FranchiseStore"], inplace=True)
    store_info.sort_values(by="NetAmount", ascending=False, inplace=True)

    final_result = get_top_channel(store_info)

    final_df = pd.merge(
        store, final_result, on=["FranchiseStore", "BusinessDate"], how="left"
    )
    final_df.to_csv("./output_files/weekly_store.csv", index=False)
    final_df = final_df[final_df["BusinessDate"] == "2025-06-08 00:00:00"]

    final_df.rename(
        columns={
            "NetAmount": "Total Revenue",
            "OrderId": "Total Orders",
            "BusinessDate": "Week",
            "FranchiseStore": "Store Id",
        },
        inplace=True,
    )
    final_df["Total Revenue"] = final_df["Total Revenue"].round(1)
    final_df[
        [
            "Week",
            "Brand",
            "Store Id",
            "Total Revenue",
            "Total Orders",
            "Average Order Value",
            "WoW Change",
            "Top2MenuItems",
            "TopOrderMethod",
        ]
    ].to_csv("./output_files/weekly_summary_sheet2.csv", index=False)
    print('Weekly summary sheet 2 created successfully!')

store_level_calculations(data)


# sheet 3
def channel_level_calculations(data):
    channel = (
        data[["NetAmount",'FranchiseStore', "OrderId", "OrderPlacedMethod"]]
        .groupby(['FranchiseStore',"OrderPlacedMethod"])
        .resample("W")
        .agg({"NetAmount": "sum", "OrderId": "nunique"})
        .reset_index()
    )

    channel.sort_values(by=["BusinessDate"], inplace=True)

    channel["Average Order Value"] = channel["NetAmount"] / channel["OrderId"]
    channel["Average Order Value"] = channel["Average Order Value"].round(2)
    channel['WoW Change'] = (
    channel.sort_values(['FranchiseStore', 'OrderPlacedMethod', 'BusinessDate'])  # Ensure proper order
      .groupby(['FranchiseStore', 'OrderPlacedMethod'])['NetAmount']
      .pct_change()  # Calculate WoW change within each group
      .multiply(100)
      .round(1)
        )
    
    channel[(channel['FranchiseStore']=='01389-00001') & (channel['OrderPlacedMethod']=='DoorDash')]
    
    channel_info = (
        data[["FranchiseStore", "MenuItemName", "OrderPlacedMethod", "NetAmount"]]
        .groupby(["FranchiseStore", "MenuItemName", "OrderPlacedMethod"])
        .resample("W")
        .agg({"NetAmount": "sum"})
        .reset_index()
    )

    channel_info.sort_values(by=["BusinessDate", "FranchiseStore",'OrderPlacedMethod'], inplace=True)
    channel_info.sort_values(by="NetAmount", ascending=False, inplace=True)
    
    def get_item_by_channel(channel_info):
        item_group = (
        channel_info.groupby(["FranchiseStore", "BusinessDate",'OrderPlacedMethod', "MenuItemName"])[
            "NetAmount"
        ]
        .sum()
        .reset_index()
        .sort_values(
            ["FranchiseStore", "BusinessDate", "NetAmount"],
            ascending=[True, True, False],
        )
    )

        item_group["ItemRank"] = item_group.groupby(["FranchiseStore",'OrderPlacedMethod', "BusinessDate"])[
            "NetAmount"
        ].rank(method="first", ascending=False)
        top2_items = item_group[item_group["ItemRank"] <= 2]

        # Combine into single string
        top2_items_agg = (
            top2_items.groupby(["FranchiseStore",'OrderPlacedMethod', "BusinessDate"])["MenuItemName"]
            .apply(lambda x: ", ".join(x))
            .reset_index(name="Top2MenuItems")
        )
        return top2_items_agg
            
 
    final_result = get_item_by_channel(channel_info)
   
    
    final_result.OrderPlacedMethod.unique()
    final_df = pd.merge(
        channel, final_result, on=["FranchiseStore",'OrderPlacedMethod' ,"BusinessDate"], how="left"
    )
    final_df.to_csv("./output_files/weekly_channel.csv", index=False)
    final_df = final_df[final_df["BusinessDate"] == "2025-06-08 00:00:00"]

    final_df.rename(
        columns={
            "NetAmount": "Total Revenue",
            "OrderId": "Total Orders",
            "BusinessDate": "Week",
            "FranchiseStore": "Store Id",
             "OrderPlacedMethod":'Digital Channel'
        },
        inplace=True,
    )
    final_df["Total Revenue"] = final_df["Total Revenue"].round(1)
    final_df['Brand']= "Little Caesars"
    
    final_df[[
            "Week",
            "Brand",
            "Store Id",
            "Digital Channel",
            "Total Revenue",
            "Total Orders",
            "Average Order Value",
            "WoW Change",
            "Top2MenuItems"]].to_csv("./output_files/weekly_summary_sheet3.csv", index=False)
    print('Weekly summary sheet 3 created successfully!')
channel_level_calculations(data)



# sheet 4
def item_level_calulation(data):
    #total revenue by dtore
    total_by_store = data.groupby(["FranchiseStore"])["NetAmount"].sum().reset_index().round(2)
    
    # total of each item by store
    item_df=data.groupby(["FranchiseStore", "MenuItemName"])["NetAmount"].sum().reset_index().sort_values(by='NetAmount', ascending=False).round(2)
    # merging total revenue by store with item df to get column of total revenue by store
    df=pd.merge(item_df, total_by_store, on='FranchiseStore', how='left', suffixes=('', '_Total'))
    # calculating revenue share percentage of each item by store
    df['RevenueSharePct'] = (df['NetAmount'] / df['NetAmount_Total'] * 100).round(2)
    df[df['FranchiseStore']=='03222-00015']['MenuItemName'].nunique()
    # filtering out items with revenue share percentage greater than 85%
    q75 = df.groupby('FranchiseStore')['RevenueSharePct'].transform(lambda x: x.quantile(0.85))
    
    # getting top items with revenue share percentage greater than 85%
    top_items = df[df['RevenueSharePct'] > q75]
    
    top_items[top_items['FranchiseStore']=='03222-00015']['MenuItemName'].nunique()
    
    # merging top items with original data to get the required columns=> transactional data
    df1=pd.merge(data.reset_index(), top_items, on=['FranchiseStore', 'MenuItemName'], how='inner')
    df1[df1['FranchiseStore']=='03222-00015']['MenuItemName'].nunique()
    

    df1.set_index('BusinessDate', inplace=True)

    # weekly aggregation of top items
    item = (
        df1[["NetAmount_x", "OrderId", "FranchiseStore",'MenuItemName','Quantity']].groupby(["FranchiseStore",'MenuItemName'])
        .resample("W")
        .agg({"NetAmount_x": "sum", "OrderId": "nunique",'Quantity': 'sum'})
        .reset_index()
    )
    
    # sorting byDate imortant  for wow chane
    item.sort_values(by=["FranchiseStore",'MenuItemName',"BusinessDate"], inplace=True)

    item["Average Order Value"] = item["NetAmount_x"] / item["OrderId"]
    item["Average Order Value"] = item["Average Order Value"].round(2)
    
    item['WoW Change'] = (
    item.sort_values(['FranchiseStore', 'MenuItemName', 'BusinessDate'])  # Ensure proper order
      .groupby(['FranchiseStore', 'MenuItemName'])['NetAmount_x']
      .pct_change()  # Calculate WoW change within each group
      .multiply(100)
      .round(1)
        )
    item[(item['FranchiseStore']=='01389-00001')& (item['MenuItemName'].isin(['3 Meat Treat','Classic Cheese']))]

    item["Brand"] = "Little Caesars"
    
    final_df = item[item["BusinessDate"] == "2025-06-08 00:00:00"]
    
    
    d=df1[df1.index >= "2025-06-02 00:00:00"]
    # their top chanel
   # Step 1: Group and sum NetAmount_x
    grouped = (
        d.groupby(['FranchiseStore', 'MenuItemName', 'OrderPlacedMethod'], as_index=False)
        ['NetAmount_x']
        .sum()
    )

    # Step 2: Sort to bring top channels to top
    sorted_grouped = grouped.sort_values(
        by=['FranchiseStore', 'MenuItemName', 'NetAmount_x'],
        ascending=[True, True, False]
    )
    
    sorted_grouped[(sorted_grouped['FranchiseStore']=='01389-00001') & (sorted_grouped['MenuItemName']=='3 Meat Treat')]

    # Step 3: Keep only the top channel per (FranchiseStore, MenuItemName)
    top_channels = sorted_grouped.drop_duplicates(
        subset=['FranchiseStore', 'MenuItemName'],
        keep='first'
    )
    top_channels.rename(columns={'OrderPlacedMethod': 'TopChannel'}, inplace=True)
    top_channels[(top_channels['FranchiseStore']=='01389-00001') & (top_channels['MenuItemName']=='3 Meat Treat')]
    
    final = pd.merge(final_df, top_channels, on=['FranchiseStore', 'MenuItemName'], how='left')
    final[(final['FranchiseStore']=='01389-00001') & (final['MenuItemName']=='3 Meat Treat')]
    
    final.columns
    final['Brand']= "LC"
    final.rename(columns={'BusinessDate':'Week',
                 'FranchiseStore':'Store Id',
                 'MenuItemName':'Product Name',
                 'Quantity':'Unit Sold',
                 'NetAmount_x_x': 'Total Revenue'}, inplace=True)
    final['Total Revenue'] = final['Total Revenue'].round(2)
    final[['Week','Store Id',
           'Product Name',
           'Unit Sold',
           'Total Revenue',
           'Average Order Value',
              'WoW Change',
              'TopChannel'
           
           ]].to_csv("./output_files/weekly_summary_sheet4.csv", index=False)
    print('Weekly summary sheet 4 created successfully!')
               
item_level_calulation(data)       

    


    


