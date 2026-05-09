#include "bcsavepathwidget.h"

#include <climits>

#include <gui/style/themecolors.h>
#include <data/crashhandler.h>

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpinBox>
#include <QLineEdit>
#include <QPushButton>
#include <QMessageBox>
#include <QDir>
#include <QFileDialog>

BCSavePathWidget::BCSavePathWidget(QWidget *parent) : QWidget(parent), SettingsStorage()
{
    auto vbl = new QVBoxLayout;

    auto lbl = new QLabel(
R"(Select the location where Blackchirp should store its files, as well as the starting experiment number. The location you choose must be writable by the current user.

When "Apply" is pressed, the directory you have chosen will be created if it does not already exist, and subdirectories will be created as well. If you choose a location where previous Blackchirp data exists, the program will attempt to determine the next available experiment number and will update the box accordingly.
)");
    lbl->setWordWrap(true);
    vbl->addWidget(lbl);

    p_lineEdit = new QLineEdit;
    auto savePath = get(BC::Key::savePath, QString(""));
    if(savePath.isEmpty())
    {
        QDir d{QDir::homePath()+"/blackchirp"};
        p_lineEdit->setText(d.absolutePath());
    }
    else
        p_lineEdit->setText(savePath);

    auto browse = new QPushButton("Browse");
    browse->setIcon(ThemeColors::createThemedIcon(":/icons/folder-open.svg", ThemeColors::IconPrimary, this));
    connect(browse, &QAbstractButton::clicked, [this](){
        auto ret = QFileDialog::getExistingDirectory(nullptr, "", QDir::homePath());
        if(!ret.isEmpty())
        {
            p_lineEdit->setText(ret);
            d_applied = false;
        }
    });

    auto hbl = new QHBoxLayout;
    hbl->addWidget(p_lineEdit, 1);
    hbl->addWidget(browse);
    vbl->addLayout(hbl);

    auto elbl = new QLabel("Starting Experiment Number");
    elbl->setAlignment(Qt::AlignRight|Qt::AlignVCenter);
    elbl->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::MinimumExpanding);

    p_expBox = new QSpinBox;
    p_expBox->setRange(1, INT_MAX);
    p_expBox->setValue(get(BC::Key::exptNum, 0) + 1);

    auto hbl2 = new QHBoxLayout;
    hbl2->addWidget(elbl, 1);
    hbl2->addWidget(p_expBox, 0);

    vbl->addLayout(hbl2);

    auto applyButton = new QPushButton("Apply");
    connect(applyButton, &QAbstractButton::clicked, this, &BCSavePathWidget::apply);
    vbl->addWidget(applyButton);

    setLayout(vbl);
}

bool BCSavePathWidget::isReady() const
{
    return d_applied;
}

void BCSavePathWidget::save()
{
    set(BC::Key::savePath, p_lineEdit->text());
    set(BC::Key::exptNum, p_expBox->value() - 1);
    SettingsStorage::save();
    CrashHandler::reopen(p_lineEdit->text());
}

void BCSavePathWidget::apply()
{
    if(p_lineEdit->text().isEmpty())
        return;

    QDir d(p_lineEdit->text());
    if(!d.exists())
    {
        if(!d.mkpath(d.absolutePath()))
        {
            QMessageBox::critical(this, "Blackchirp Error", QString("Could not create %1.\nPlease choose a different location.").arg(d.absolutePath()));
            return;
        }
    }

    if(!d.cd(BC::Key::exptDir))
    {
        if(!d.mkpath(BC::Key::exptDir))
        {
            QMessageBox::critical(this, "Blackchirp Error", QString("Could not create experiments directory in %1.\nPlease choose a different location.").arg(d.absolutePath()));
            return;
        }

        if(!d.mkpath(BC::Key::logDir))
        {
            QMessageBox::critical(this, "Blackchirp Error", QString("Could not create log directory in %1.\nPlease choose a different location.").arg(d.absolutePath()));
            return;
        }

        if(!d.mkpath(BC::Key::exportDir))
        {
            QMessageBox::critical(this, "Blackchirp Error", QString("Could not create exports directory in %1.\nPlease choose a different location.").arg(d.absolutePath()));
            return;
        }

        if(!d.mkpath(BC::Key::trackingDir))
        {
            QMessageBox::critical(this, "Blackchirp Error", QString("Could not create tracking data directory in %1.\nPlease choose a different location.").arg(d.absolutePath()));
            return;
        }

        p_expBox->setValue(1);
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
                max = qMax(i, max);
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
                    max = qMax(i, max);
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
                        max = qMax(i, max);
                }

                p_expBox->setValue(max + 1);
            }
        }
    }

    d_applied = true;
    emit applied();
}
