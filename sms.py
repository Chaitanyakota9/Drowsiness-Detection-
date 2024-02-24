from twilio.rest import Client

account_sid = 'ACef5d7bbdb28013a19c88f7874d4f1ada'
auth_token = '5deca3ead6cd4d399aa281cbcb47040a'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                     body="Stop the vehicle with number:8114",
                     from_='+12315254666',
                     to='+918919859493'
                 )
print(message.sid)