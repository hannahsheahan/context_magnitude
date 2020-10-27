myaddress = 'data2server4storage@gmail.com'; % fill in email address
mypassword = 'd4t4stor4ge'; % fill in password

setpref('Internet','E_mail',myaddress);
setpref('Internet','SMTP_Server','smtp.gmail.com');
setpref('Internet','SMTP_Username',myaddress);
setpref('Internet','SMTP_Password',mypassword);

props = java.lang.System.getProperties;
props.setProperty('mail.smtp.auth','true');
props.setProperty('mail.smtp.socketFactory.class', ...
    'javax.net.ssl.SSLSocketFactory');
props.setProperty('mail.smtp.socketFactory.port','465');

allfiles        = dir(fullfile(datafolder,'Con_ppt_*.mat')); % send which file? Looks for most recent
[blah,dx]       = sort([allfiles.datenum]);
relevantFile    = allfiles(dx(end)).name;

sendmail(myaddress, relevantFile, 'Numbers\n',{relevantFile}); % fill in title of mail