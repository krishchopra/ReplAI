docker pull tyrrrz/discordchatexporter
sudo docker run tyrrrz/discordchatexporter exportdm -t YOUR_DISCORD_TOKEN -f Json
docker cp <container_id>:/out data/raw/discord
docker rm <container_id>