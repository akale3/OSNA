"""
collect.py
"""
import datetime
import json
import os
import sys
import time
from pathlib import Path

from TwitterAPI import TwitterAPI
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

"""All Access Keys"""
consumer_key = 'aCHgwVfyQXuPHTG6nsX5zYQzl'
consumer_secret = 'qjdIYTGcoq9xbZRVDYT33iwiWcWI1LaNbc4vapEKT9c7ab2Lz6'
access_token = '767529744-ySmXCWGSciC0rdwk6eX89SWJBZaGk4zgvECansaJ'
access_token_secret = 'jPIwvOUSdKxMOZY3taEETjYnwFTpsxjV1oXZ87FAFFuFq'

"""Creating Authentication Handler"""
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)


def deleteFiles(folderPath):
    for file in os.listdir(folderPath):
        filePath = Path(folderPath + file)
        try:
            if filePath.is_file():
                os.remove(folderPath + file)
        except Exception as e:
            print(e)


def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


class ApplicationStreamListener(StreamListener):
    def __init__(self):
        super(ApplicationStreamListener, self).__init__()
        self.tweetCount = 0

    def on_data(self, data):
        parsedData = json.loads(data)
        if 'text' in parsedData:
            tweetText = parsedData['text']
            if not tweetText.startswith('rt') | tweetText.startswith('RT'):
                if self.tweetCount < 200:
                    try:
                        with open('data/iphone.txt', 'a') as outfile:
                            outfile.write(data)
                            outfile.write('\n')
                            self.tweetCount += 1
                        return True
                    except BaseException as e:
                        print("Error on_status: %s" % str(e))
                else:
                    endTime = datetime.datetime.now()
                    print(endTime - startTime)
                    return False

    def on_error(self, status):
        print(status)


def getTweets(stream, hashTag):
    stream.filter(track=[hashTag], languages=['en'])


def get_users_friends():
    users = list()
    f = open('data/iphone.txt', 'r')
    for line in f:
        if len(line) > 1:
            users.append(json.loads(line)['user']['screen_name'])

    for screenName in users:
        request = robust_request(get_twitter(), 'friends/ids',
                                 {'screen_name': screenName, 'count': 5000})
        with open('data/iphoneUsersFriends.txt', 'a') as friendsFile:
            friendList = json.loads(request.text)["ids"][:5000]
            if len(friendList) > 0:
                user = dict()
                user['screen_name'] = screenName
                user['friend_ids'] = friendList
                userData = json.dumps(user)
                friendsFile.write(userData)
                friendsFile.write('\n')


startTime = datetime.datetime.now()


def main():
    """Deleting existing data"""
    folderPath = 'data/'
    deleteFiles(folderPath)

    """Create a stream to get tweets"""
    streamListener = ApplicationStreamListener()
    stream = Stream(auth, streamListener)

    """Get tweets from stream"""
    hashTag = 'iphone'
    getTweets(stream, hashTag)

    """Get collected tweets users and retrive there friends"""
    get_users_friends()


if __name__ == '__main__':
    main()
