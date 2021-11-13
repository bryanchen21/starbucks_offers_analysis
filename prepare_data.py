import pandas as pd
import numpy as np
import datetime


def process_profile(df):
    ''' Calculate membership years and remove rows with missing gender
        INPUT: profile df

        OUTPUT: profile df
    '''
    df.rename(columns={'id': 'person'}, inplace=True)

    # There are 2175 rows where there is no gender, income and age is 118.
    # This is likely to be errors in the data as an age of 118 is highly unlikely.
    # These are removed from the data
    profile = df[~df.gender.isna()]

    profile['became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d')
    profile['membership_years'] = profile['became_member_on'].apply(lambda x: (datetime.datetime.today() - x).days / 365)

    return profile

def process_portfolio(df):
    ''' Using one hot encoding, we create dummy variables for labels in channels column. 4 dummy variables are created
        INPUT: portfolio df

        OUTPUT: portfolio df
    '''

    # Extract labels from channels column and create dummy variables based on labels
    channel_cat_df = df[['channels']]
    new_cols = channel_cat_df.iloc[1, :][0]
    channel_cat_df = channel_cat_df.reindex(columns=channel_cat_df.columns.tolist() + new_cols)

    for column in channel_cat_df.columns[1:]:
        value_list = []

        for x in channel_cat_df["channels"].tolist():
            if column in x:
                value_list.append(1)

            else:
                value_list.append(0)
        channel_cat_df[column] = value_list
    channel_cat_df.drop(['channels'], axis=1, inplace=True)

    df = pd.concat([df, channel_cat_df], axis=1)

    df.rename(columns={"id": "offer_id"}, inplace=True)

    return df

def process_transcript(transcript, profile):
    ''' Remove from data rows related to customers that are not in the demographic dataset.
        Extract offer info and transaction amount from "value" column and append as additional columns.
        Separate df into offers received, offers viewed, offers completed and transactions.
        INPUT: transcript df
               profile df

        OUTPUT: offers_received_df
                offers_viewed_df
                offers_completed_df
                txn_df
    '''

    # Remove 33772 rows related to people that are not in profile table
    # This consists of 2175 individuals
    transcript = transcript[transcript.person.isin(list(profile.person))]

    # "Value" column contains dictionary of offer id or transaction amount
    # Extract offer id and transaction amount and populate in offer id column and amount column
    value_cat_df = transcript[['value']]

    dict_key_full_list = []

    for item in list(transcript.value):
        dict_key_full_list.append(list(item.keys())[0])
    new_cols = list(set(dict_key_full_list))

    value_cat_df = value_cat_df.reindex(columns=value_cat_df.columns.tolist() + new_cols)

    for column in value_cat_df.columns[1:]:
        value_list = []

        for x in value_cat_df["value"].tolist():
            if column in x.keys():
                value_list.append(x[column])

            else:
                value_list.append(np.nan)
        value_cat_df[column] = value_list

    value_cat_df.drop(['value'], axis=1, inplace=True)
    transcript = pd.concat([transcript, value_cat_df], axis=1)

    transcript.offer_id.fillna(transcript['offer id'], inplace=True)

    transcript.drop(['offer id', "value"], axis=1, inplace=True)

    transcript.sort_values(by=["person", 'time'], inplace=True)

    transcript["timedays"] = transcript.time / 24

    #Extract records related to transactions
    txn_df = transcript[transcript['event'].str.contains("transaction")].drop(["offer_id", "time"], axis=1).sort_values(
        by=["person", 'timedays'])
    txn_df.rename(columns={"timedays": "txnday"}, inplace=True)

    # Extract offer-related records
    offers_raw = transcript[transcript['event'].str.contains("offer")].drop(["amount","time"],axis=1)
    # Create dummies for whether offer received, viewed or completed
    offer_dummies = pd.get_dummies(offers_raw.event)
    offers_raw = pd.concat([offers_raw, offer_dummies], axis=1)

    # Split offers into received, viewed or completed
    offers_received_df = offers_raw[offers_raw["offer received"] == 1].reset_index(drop=True)
    offers_viewed_df = offers_raw[offers_raw["offer viewed"] == 1].reset_index(drop=True)
    offers_completed_df = offers_raw[offers_raw["offer completed"] == 1].reset_index(drop=True)


    return offers_received_df, offers_viewed_df, offers_completed_df, txn_df

def create_merged_df(transcript,profile, portfolio ):
    ''' Create a master data set containing:
        a) person.
        b) offer received.
        c) offer received day.
        d) offer expiry day.
        e) whether the offer was viewed.
        f) whether the offer was completed.
        g) offer completion day.
        h) transaction amount
        i) whether the offer worked
        j) amount that customer needs to spend for the offer.
        k) reward that customer gets from the offer.
        l) 3 dummy variables for whether the offer is bogo, discount or informational.
        m) 3 dummy variables for gender.
        n) 4 dummy variables for whether the offer is through web, email, mobile, social media.
        o) age.
        p) income.
        q) duration of membership in years.
        INPUT: transcript df
               profile df
               portfolio df
        OUTPUT: merged_df
    '''
    offers_received_df, offers_viewed_df, offers_completed_df, txn_df = process_transcript(transcript, profile)

    merged_df = pd.DataFrame()
    for i in range(len(offers_received_df)):
        # Get offer id
        sel_offer_id = offers_received_df.iloc[i]["offer_id"]
        # Get customer id
        customerid = offers_received_df.iloc[i]["person"]
        # Get transactions for this customer
        customer_txn_df = txn_df[txn_df["person"] == customerid]
        # Get day that order is received and calculate the date the offer ends
        offer_received_day = offers_received_df.iloc[i]["timedays"]
        duration = portfolio[portfolio["offer_id"] == sel_offer_id]["duration"].values[0]
        offer_end_day = offer_received_day + duration

        # Get df containing list of views and completions for this customer and offer id
        customer_viewed_df = offers_viewed_df[
            (offers_viewed_df["person"] == customerid) & (offers_viewed_df["offer_id"] == sel_offer_id)]
        customer_completed_df = offers_completed_df[
            (offers_completed_df["person"] == customerid) & (offers_completed_df["offer_id"] == sel_offer_id)]

        # Handle cases where offer is received more than once by the same person, and thus can be viewed more than once.
        # If a person receives 2 of the same offers on different days and did view them both, to determine which view timestamp belongs
        # to which offer, we take the timedays of both views and see which view is closest to the offer received day
        num_views = len(customer_viewed_df)

        if num_views > 1:
            diff_list = []
            for idx, row in customer_viewed_df.iterrows():
                diff_list.append(abs(row["timedays"] - offer_received_day))

            val, idx = min((val, idx) for (idx, val) in enumerate(diff_list))
            customer_viewed_df = customer_viewed_df.iloc[[idx]]
        # Handle cases where offer is received more than once by the same person, and thus can be completed more than once
        # If a person receives 2 of the same offers and did complete them both, we take the timedays of both completion and see
        # which completion is closest to the offer received day
        num_completed = len(customer_completed_df)

        if num_completed > 1:
            diff_list = []
            for idx, row in customer_completed_df.iterrows():
                diff_list.append(abs(row["timedays"] - offer_received_day))

            val, idx = min((val, idx) for (idx, val) in enumerate(diff_list))

            customer_completed_df = customer_completed_df.iloc[[idx]]

        # Whether viewed or not
        # An offer is considered "viewed" if it is viewed before it is completed
        if (len(customer_viewed_df) > 0) & (len(customer_completed_df) > 0):

            if (customer_viewed_df.timedays.values[0] <= customer_completed_df.timedays.values[0]) & (
                    customer_viewed_df.timedays.values[0] >= offer_received_day):
                viewed = 1
            else:
                viewed = 0

        elif (len(customer_viewed_df) > 0) & (len(customer_completed_df) == 0):
            if (customer_viewed_df.timedays.values[0] <= offer_end_day):
                viewed = 1
            else:
                viewed = 0

        else:
            viewed = 0

        # Whether completed or not
        # An offer is considered "completed" if it is completed before the offer end day.
        # To overcome instances where offer is received twice but only completed for one of them
        if len(customer_completed_df) > 0:

            if (customer_completed_df.timedays.values[0] <= offer_end_day):
                completed = 1
            else:
                completed = 0

        else:
            completed = 0

        if completed == 1:
            offer_completed_day = customer_completed_df.timedays.values[0]

            # Get transaction amount for each offer
            sel_txn = np.logical_and(customer_txn_df['txnday'] >= offer_received_day,
                                     customer_txn_df['txnday'] <= offer_completed_day)
            amount_spent = sum(customer_txn_df[sel_txn]['amount'].values)
        else:
            offer_completed_day = np.nan
            sel_txn = np.logical_and(customer_txn_df['txnday'] >= offer_received_day,
                                     customer_txn_df['txnday'] <= offer_end_day)
            amount_spent = sum(customer_txn_df[sel_txn]['amount'].values)
        # Create dataframe for unique customer, offer id and offer received day
        data = pd.DataFrame([{
            "person": customerid,
            "offer_id": sel_offer_id,
            "offer_received_day": offer_received_day,
            "offer_end_day": offer_end_day,
            "offer_viewed": viewed,

            "offer_completed": completed,
            "offer_completed_day": offer_completed_day,
            "transaction": amount_spent}])
        merged_df = pd.concat([merged_df, data], axis=0)

    return merged_df