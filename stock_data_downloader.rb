#require "websocket-native"
require "pusher-client"
require 'json'

puts "starting client"
client= PusherClient::Socket.new("de504dc5763aeef9ff52")
client.connect(true)
PusherClient.logger.level= Logger::Severity::WARN

puts "subscribing to channels"
live_trades= client.subscribe("live_trades")
order_book= client.subscribe("order_book")


price = 0.0
amount = 0.0

puts "binding events"
order_book.bind("data") do |data|
  json = JSON.parse(data)
  timestamp = json["timestamp"]
  if amount > 0 
	max = 19
	json["bids"] = json["bids"][0..max]
	json["asks"] = json["asks"][0..max]
  	json["price"] = price
  	json["amount"] = amount
  	File.write("stock_data/#{timestamp}.json", JSON.generate(json))
  end
end

puts "binding events"
live_trades.bind("trade") do |data|
  json = JSON.parse(data)
  price = json["price"]
  amount = json["amount"]
end

puts "waiting for data"
while true do
  sleep 1
end


