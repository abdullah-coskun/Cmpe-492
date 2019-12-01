import boto3
from botocore.exceptions import ClientError
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart

from rest_framework.exceptions import ValidationError


def send_email_results(email,file):


    SENDER = "Classify <abdullah.coskun1@boun.edu.tr>"

    # Replace recipient@example.com with a "To" address. If your account
    # is still in the sandbox, this address must be verified.
    RECIPIENT = email


    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "eu-west-1"

    # The subject line for the email.
    SUBJECT = "Results of classification"

    # The full path to the file that will be attached to the email.
    try:
        ATTACHMENT = file.temporary_file_path()
    except:
        ATTACHMENT=""

    f = open("classify/aws_key.txt", "r")
    AWS_KEY = f.readline().rstrip()
    AWS_SECRET_KEY = f.readline().rstrip()
    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = "Hello,\r\nRequested results of classification is in attachment."

    # The HTML body of the email.
    #BODY_HTML = feedback_email_part1 + feedback_instance["name"] + email_and_phone_number + feedback_email_part2 + feedback_instance["message"] + feedback_email_part3


    # The character encoding for the email.
    CHARSET = "utf-8"

    # Create a new SES resource and specify a region.
    client = boto3.client('ses', region_name=AWS_REGION, aws_access_key_id=AWS_KEY,
                          aws_secret_access_key=AWS_SECRET_KEY)

    # Create a multipart/mixed parent container.
    msg = MIMEMultipart('mixed')
    # Add subject, from and to lines.
    msg['Subject'] = SUBJECT
    msg['From'] = SENDER
    msg['To'] = RECIPIENT

    msg_body = MIMEMultipart('alternative')

    textpart = MIMEText(BODY_TEXT.encode(CHARSET), 'plain', CHARSET)
    #htmlpart = MIMEText(BODY_HTML.encode(CHARSET), 'html', CHARSET)

    # Add the text and HTML parts to the child container.
    msg_body.attach(textpart)
    #msg_body.attach(htmlpart)
    filename = file.name
    if ATTACHMENT is "":
        att = MIMEApplication(file.read())
        att.add_header('Content-Disposition', 'attachment', filename=filename)
    else:
        att = MIMEApplication(open(ATTACHMENT, 'rb').read())
        att.add_header('Content-Disposition', 'attachment', filename=filename)
    msg.attach(msg_body)

    # Add the attachment to the parent container.
    msg.attach(att)
    # print(msg)
    try:
        # Provide the contents of the email.
        response = client.send_raw_email(
            Source=SENDER,
            Destinations=[
                RECIPIENT
            ],
            RawMessage={
                'Data': msg.as_string(),
            },
        )
    # Display an error if something goes wrong.
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])
