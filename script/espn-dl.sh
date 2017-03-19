#!/bin/bash


set -e
SPORT="nfl"
IMG_DIR="img"

get_teams() {
    curl "http://www.espn.com/$SPORT/teams" 2>/dev/null |
        grep -o "http://www.espn.com/$SPORT/team/_/name/[a-z]\{1,\}/" |
        sort | uniq | cut -d'/' -f8
}

get_players() {
    TEAM="$1"
    curl "http://www.espn.com/$SPORT/team/roster/_/name/$TEAM" 2>/dev/null |
        grep -o "http://www.espn.com/$SPORT/player/_/id/[0-9]\{1,\}/" |
        sort | uniq | cut -d'/' -f8
}

download_player_image() {
    PLAYER_ID="$1"
    curl "http://a.espncdn.com/combiner/i?img=/i/headshots/$SPORT/players/full/$PLAYER_ID.png" >"$IMG_DIR/$SPORT-$PLAYER_ID"
}

loop_players() {
    cat - | while read PLAYER_ID; do
        download_player_image "$PLAYER_ID"
        sleep 1
    done
}

loop_teams() {
    cat - | while read TEAM; do
        get_players "$TEAM" | loop_players
        sleep 10
    done
}

mkdir -p "$IMG_DIR"
get_teams | loop_teams

