"""Gmail integration helpers."""

import os
import base64

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly'
]


def _import_google_libraries():

    try:

        from google_auth_oauthlib.flow import InstalledAppFlow

        from googleapiclient.discovery import build

        return InstalledAppFlow, build

    except ImportError as exc:

        raise ImportError(
            'Install: google-auth-oauthlib google-api-python-client'
        ) from exc


def build_gmail_service(
    credentials_path='credentials.json'
):

    InstalledAppFlow, build = _import_google_libraries()

    if not os.path.exists(credentials_path):

        raise FileNotFoundError(
            f'Credentials file not found: {credentials_path}'
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        credentials_path,
        SCOPES
    )

    creds = flow.run_local_server(port=0)

    service = build(
        'gmail',
        'v1',
        credentials=creds
    )

    return service


def list_unread_messages(
    service,
    max_results=10
):

    results = service.users().messages().list(
        userId='me',
        q='in:anywhere newer_than:7d',
        maxResults=max_results
    ).execute()

    return results.get('messages', [])


def get_message_payload(
    service,
    message_id
):

    message = service.users().messages().get(
        userId='me',
        id=message_id,
        format='full'
    ).execute()

    return message


def extract_email_body(payload):

    try:

        parts = payload['payload'].get(
            'parts',
            []
        )

        for part in parts:

            if part['mimeType'] == 'text/plain':

                data = part['body'].get('data')

                if data:

                    decoded = base64.urlsafe_b64decode(
                        data
                    )

                    return decoded.decode(
                        'utf-8',
                        errors='ignore'
                    )

        body_data = payload['payload']['body'].get(
            'data'
        )

        if body_data:

            decoded = base64.urlsafe_b64decode(
                body_data
            )

            return decoded.decode(
                'utf-8',
                errors='ignore'
            )

    except Exception as e:

        print(
            'EMAIL PARSE ERROR:',
            str(e)
        )

    return ''