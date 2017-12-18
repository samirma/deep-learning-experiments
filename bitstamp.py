#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pusherclient #live stream client: https://github.com/ekulyk/PythonPusherClient
import logging
import time
import json

global current_state


pusher = {}

trade_file = '/tmp/trade.json'
state_file = '/tmp/state.json'


def save_trade(data):
    with open(trade_file, 'w') as file:
        file.write(data)
        
def load_trade():
    trades_file = open(trade_file,'r')
    last_trade = json.loads(trades_file.read())
    trades_file.close
    return last_trade

      
def save_state(data):
    with open(state_file, 'w') as file:
        file.write(json.dumps(data))
        
def load_state():
    trades_file = open(state_file,'r')
    last_trade = json.loads(trades_file.read())
    trades_file.close
    return last_trade
      
    
    
def trade_callback(data): 
        save_trade(data)

def order_book_callback(data):
    current_state = json.loads(data)
    orders_limit = 10
    current_state["bids"] = current_state["bids"][:orders_limit]
    current_state["asks"] = current_state["asks"][:orders_limit]
    last_trade = load_trade()
    current_state["amount"] = last_trade["amount"]
    current_state["price"] = last_trade["price"]
    save_state(current_state)
    

class BitStamp:

    def start(self):
        
        def connect_handler(data):
            print("connect_handler")
            trades_channel = pusher.subscribe("live_trades")
            trades_channel.bind('trade', trade_callback)
            order_book_channel = pusher.subscribe('order_book');
            order_book_channel.bind('data', order_book_callback)
        
        pusher = pusherclient.Pusher("de504dc5763aeef9ff52")
        pusher.connection.bind('pusher:connection_established', connect_handler)
        pusher.connect()
        self.pusher = pusher
        
        #while True:
        #    time.sleep(3)
        
    def stop(self):
        self.pusher.disconnect()
        
    def get_current_state(self):
        return load_state()
