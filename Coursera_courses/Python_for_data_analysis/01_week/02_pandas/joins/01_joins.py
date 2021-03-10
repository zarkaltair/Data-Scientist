import pandas as pd


calendar = pd.read_csv('calendar_sept.csv', index_col=0)
reviews = pd.read_csv('reviews.csv', index_col=0)
listings = pd.read_csv('listings.csv', index_col=0)

calendar.head(2)
reviews.head(2)
listings.head(2)

print(pd.merge(listings, reviews, left_on=['id'], right_on=['listing_id'], how='right', indicator=True))

calendar.set_index('listing_id', inplace=True)
print(calendar.join(reviews, lsuffix='listing_id', rsuffix='listing_id'))
