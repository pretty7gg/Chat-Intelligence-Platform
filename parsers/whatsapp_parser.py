"""
whatsapp_parser.py

This is your original preprocessor.py logic, restructured to fit the
BaseChatParser contract. The parsing logic itself (regex splitting,
date extraction) is unchanged from your original project — we're only
reorganizing it so a future TelegramParser or SlackParser could sit
right next to this one without touching app.py.
"""

import re
import pandas as pd
from parsers.base import BaseChatParser


class WhatsAppParser(BaseChatParser):
    """Parses a raw exported WhatsApp .txt chat file."""

    # Matches WhatsApp's date/time prefix, e.g. "12/08/2025, 14:32 - "
    DATE_PATTERN = r"\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s"

    def parse(self, raw_text: str) -> pd.DataFrame:
        messages = re.split(self.DATE_PATTERN, raw_text)[1:]
        dates = re.findall(self.DATE_PATTERN, raw_text)

        df = pd.DataFrame({"user_message": messages, "message_date": dates})

        df["message_date"] = pd.to_datetime(
            df["message_date"], format="%d/%m/%Y, %H:%M - "
        )
        df.rename(columns={"message_date": "date"}, inplace=True)

        users = []
        cleaned_messages = []
        for message in df["user_message"]:
            # A real message looks like "Rahul: hey what's up"
            # A system message (like "X joined") has no "name: " prefix
            entry = re.split(r"([\w\W]+?):\s", message)
            if entry[1:]:
                users.append(entry[1])
                cleaned_messages.append(" ".join(entry[2:]))
            else:
                users.append("group_notification")
                cleaned_messages.append(entry[0])

        df["user"] = users
        df["message"] = cleaned_messages
        df.drop(columns=["user_message"], inplace=True)

        # Drop system/group notifications - not real conversation content
        df = df[df["user"] != "group_notification"]

        # Derived time columns - kept because your existing stats/timeline
        # features (monthly timeline, activity heatmap, etc.) rely on these
        df["only_date"] = df["date"].dt.date
        df["year"] = df["date"].dt.year
        df["month_num"] = df["date"].dt.month
        df["month"] = df["date"].dt.month_name()
        df["day"] = df["date"].dt.day
        df["day_name"] = df["date"].dt.day_name()
        df["hour"] = df["date"].dt.hour
        df["minute"] = df["date"].dt.minute

        period = []
        for hour in df["hour"]:
            if hour == 23:
                period.append("23-00")
            elif hour == 0:
                period.append("00-1")
            else:
                period.append(f"{hour}-{hour + 1}")
        df["period"] = period

        return self.validate(df)
