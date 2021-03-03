from __future__ import absolute_import, division, unicode_literals

import logging


from rasa_sdk import Action
from rasa_sdk.events import SlotSet
from utils.city_check import check_location
from utils.email_cred_config import Config
from utils.mail_service import send_email
from utils.zomato_slots import results

logging.basicConfig(level="DEBUG")


class ActionSearchRestaurants(Action):
    def name(self):
        return 'action_restaurant'

    def run(self, dispatcher, tracker, domain):
        loc = tracker.get_slot('location')
        cuisine = tracker.get_slot('cuisine')
        price = tracker.get_slot('price')

        global restaurants

        restaurants = results(loc, cuisine, price)
        restaurants.drop_duplicates(inplace=True)
        top5 = restaurants.head(5)

        # top 5 results to display
        if len(top5) > 0:
            response = 'Showing you top results:' + "\n"
            for index, row in top5.iterrows():
                response = response + str(row["restaurant_name"]) + ' (rated ' + row['restaurant_rating'] + ') in ' + row['restaurant_address'] + \
                    ' and the average budget for two people ' + str(row['budget_for2people'])+"\n"
        else:
            response = 'No restaurants found'

        dispatcher.utter_message(str(response))


class SendMail(Action):
    def name(self):
        return 'email_restaurant_details'

    def run(self, dispatcher, tracker, domain):
        recipient = tracker.get_slot('email')

        top10 = restaurants.head(10)
        send_email(recipient, top10)

        dispatcher.utter_message("Have a great day!")


class Check_location(Action):
    def name(self):
        return 'action_check_location'

    def run(self, dispatcher, tracker, domain):
        loc = tracker.get_slot('location')
        check = check_location(loc)

        return [SlotSet('location', check['location_new']), SlotSet('location_found', check['location_f'])]
