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
timestamp = 0.0

puts "binding events"
order_book.bind("data") do |data|
  json = JSON.parse(data)
  if amount > 0 
  	json["price"] = price
  	json["amount"] = amount
  	json["timestamp"] = timestamp
  	File.write("#{timestamp}.json", JSON.pretty_generate(json))
  end
end

puts "binding events"
live_trades.bind("trade") do |data|
  json = JSON.parse(data)
  price = json["price"]
  amount = json["amount"]
  timestamp = json["timestamp"]
end

puts "waiting for data"
while true do
  sleep 1
end


