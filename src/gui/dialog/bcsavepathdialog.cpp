#include "bcsavepathdialog.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QDialogButtonBox>
#include <QMessageBox>
#include <QDir>
#include <QFileDialog>

BCSavePathDialog::BCSavePathDialog(QWidget *parent) : QDialog(parent), SettingsStorage()
{
    auto vbl = new QVBoxLayout;

    auto lbl = new QLabel(
R"(Select the location where Blackchirp should store its files, as well as the starting experiment number. The location you choose must be writable by the current user.

When "Apply" is pressed, the directory you have chosen will be created if it does not already exist, and subdirectories will be created as well. If you choose a location where previous Blackchirp data exists, the program will attempt to determine the next available experiment number and will update the box accordingly.
)");
    lbl->setWordWrap(true);
    vbl->addWidget(lbl);

    p_lineEdit = new QLineEdit;
    auto savePath = get(BC::Key::savePath,QString(""));
    if(savePath.isEmpty())
    {
        QDir d{QDir::homePath()+"/blackchirp"};
        p_lineEdit->setText(d.absolutePath());
    }
    else
        p_lineEdit->setText(savePath);

    auto browse = new QPushButton("Browse");
    connect(browse,&QAbstractButton::clicked,[this](){
        auto ret = QFileDialog::getExistingDirectory(nullptr,"",QDir::homePath());
        if(!ret.isEmpty())
        {
            p_lineEdit->setText(ret);
            p_buttons->button(QDialogButtonBox::Save)->setEnabled(false);
        }
    });

    auto hbl = new QHBoxLayout;
    hbl->addWidget(p_lineEdit,1);
    hbl->addWidget(browse);
    vbl->addLayout(hbl);

    auto elbl = new QLabel("Starting Experiment Number");
    elbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
    elbl->setSizePolicy(QSizePolicy::Minimum,QSizePolicy::MinimumExpanding);

    p_expBox = new QSpinBox;
    p_expBox->setRange(1,__INT_MAX__);
    p_expBox->setValue(get(BC::Key::exptNum,0)+1);

    auto hbl2 = new QHBoxLayout;
    hbl2->addWidget(elbl,1);
    hbl2->addWidget(p_expBox,0);

    vbl->addLayout(hbl2);

    p_buttons = new QDialogButtonBox(QDialogButtonBox::Save|QDialogButtonBox::Apply|QDialogButtonBox::Cancel);
    p_buttons->button(QDialogButtonBox::Save)->setEnabled(false);

    connect(p_buttons->button(QDialogButtonBox::Cancel),&QAbstractButton::clicked,
            this,&BCSavePathDialog::reject);
    connect(p_buttons->button(QDialogButtonBox::Apply),&QAbstractButton::clicked,
            this,&BCSavePathDialog::apply);
    connect(p_buttons->button(QDialogButtonBox::Save),&QAbstractButton::clicked,
            this,&BCSavePathDialog::accept);

    vbl->addWidget(p_buttons);
    setLayout(vbl);
}


void BCSavePathDialog::accept()
{
    set(BC::Key::savePath,p_lineEdit->text());
    set(BC::Key::exptNum,p_expBox->value()-1);
    save();
    return QDialog::accept();
}

void BCSavePathDialog::reject()
{
    auto sp = get(BC::Key::savePath,QString(""));
    auto en = get(BC::Key::exptNum,0)+1;
    if(sp.isEmpty())
    {
        int ret = QMessageBox::critical(this,"Blackchirp Error",QString("You must select a valid directory in order to run Blackchirp. Would you like to try again?"),QMessageBox::No|QMessageBox::Yes,QMessageBox::Yes);
        if(ret == QMessageBox::No)
            QDialog::reject();
    }
    else
    {
        int ret = QMessageBox::warning(this,"Blackchirp Warning",QString("The current save path is set to %1 and the current experiment number is %2. These values will not be changed.\n\nIs this correct?").arg(sp).arg(en),QMessageBox::No|QMessageBox::Yes,QMessageBox::Yes);
        if(ret == QMessageBox::Yes)
            QDialog::reject();
    }
}

void BCSavePathDialog::apply()
{
    if(p_lineEdit->text().isEmpty())
        return;

    QDir d(p_lineEdit->text());
    if(!d.exists())
    {
        if(!d.mkpath(d.absolutePath()))
        {
            QMessageBox::critical(this,"Blackchirp Error",QString("Could not create %1.\nPlease choose a different location.").arg(d.absolutePath()));
            return;
        }
    }

    if(!d.cd(BC::Key::exptDir))
    {
        if(!d.mkpath(BC::Key::exptDir))
        {
            QMessageBox::critical(this,"Blackchirp Error",QString("Could not create experiments directory in %1.\nPlease choose a different location.").arg(d.absolutePath()));
            return;
        }

        if(!d.mkpath(BC::Key::logDir))
        {
            QMessageBox::critical(this,"Blackchirp Error",QString("Could not create log directory in %1.\nPlease choose a different location.").arg(d.absolutePath()));
            return;
        }

        if(!d.mkpath(BC::Key::exportDir))
        {
            QMessageBox::critical(this,"Blackchirp Error",QString("Could not create exports directory in %1.\nPlease choose a different location.").arg(d.absolutePath()));
            return;
        }

        if(!d.mkpath(BC::Key::trackingDir))
        {
            QMessageBox::critical(this,"Blackchirp Error",QString("Could not create tracking data directory in %1.\nPlease choose a different location.").arg(d.absolutePath()));
            return;
        }

        p_expBox->setValue(1);
        p_buttons->button(QDialogButtonBox::Save)->setEnabled(true);
    }
    else
    {
        //search for previous experiments and update experiment number
        auto l1 = d.entryList(QDir::Dirs);
        int max = -1;
        for(auto &s : l1)
        {
            bool ok = false;
            auto i = s.toInt(&ok);
            if(ok)
                max = qMax(i,max);
        }

        if(max >= 0)
        {
            d.cd(QString::number(max));
            auto l2 = d.entryList(QDir::Dirs);
            max = -1;
            for(auto &s : l2)
            {
                bool ok = false;
                auto i = s.toInt(&ok);
                if(ok)
                    max = qMax(i,max);
            }

            if(max >= 0)
            {
                d.cd(QString::number(max));
                auto l3 = d.entryList(QDir::Dirs);
                max = -1;
                for(auto &s : l3)
                {
                    bool ok = false;
                    auto i = s.toInt(&ok);
                    if(ok)
                        max = qMax(i,max);
                }

                p_expBox->setValue(max+1);
            }
        }
    }

    p_buttons->button(QDialogButtonBox::Save)->setEnabled(true);
}


QSize BCSavePathDialog::sizeHint() const
{
    return {400,300};
}
