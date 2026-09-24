"""One-time Google OAuth consent, run LOCALLY, prints the refresh token.

    python -m live.close_signal.auth_once --client-id ... --client-secret ...

Create an OAuth client of type "Desktop app" in Google Cloud Console with the
Calendar API enabled, run this once, approve the consent screen in the browser,
and paste the printed refresh token into the repository secret
``GCAL_REFRESH_TOKEN`` (client id and secret into ``GCAL_CLIENT_ID`` /
``GCAL_CLIENT_SECRET``, the target calendar's id into ``GCAL_CALENDAR_ID``;
``primary`` is your main calendar).  Nothing is stored on disk here.
"""

from __future__ import annotations

import argparse

from live.close_signal.calendar_push import SCOPES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client-id", required=True)
    ap.add_argument("--client-secret", required=True)
    a = ap.parse_args()
    from google_auth_oauthlib.flow import InstalledAppFlow  # type: ignore

    flow = InstalledAppFlow.from_client_config(
        {
            "installed": {
                "client_id": a.client_id,
                "client_secret": a.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": ["http://localhost"],
            }
        },
        scopes=list(SCOPES),
    )
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
    print("GCAL_REFRESH_TOKEN=" + str(creds.refresh_token))


if __name__ == "__main__":
    main()
