import smtplib

pw = "*"
email_address = "*"
def notify(message="python script done"):
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(email_address, pw)
    msg = "\r\n".join([
        "From: " + email_address,
        "To: " + email_address,
        "Subject: " + message,
        "",
        message
    ])

    server.sendmail(email_address, email_address, msg)
    server.quit()