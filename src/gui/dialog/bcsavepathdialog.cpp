#include "bcsavepathdialog.h"
#include <gui/widget/bcsavepathwidget.h>
#include <gui/style/themecolors.h>
#include <data/storage/settingsstorage.h>

#include <QVBoxLayout>
#include <QDialogButtonBox>
#include <QPushButton>
#include <QMessageBox>

BCSavePathDialog::BCSavePathDialog(QWidget *parent) : QDialog(parent)
{
    setWindowIcon(ThemeColors::createThemedIcon(":/icons/bc_logo_trans.svg", ThemeColors::IconPrimary, this));

    auto vbl = new QVBoxLayout;

    p_widget = new BCSavePathWidget(this);
    vbl->addWidget(p_widget);

    p_buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel);
    p_buttons->button(QDialogButtonBox::Save)->setEnabled(false);

    connect(p_widget, &BCSavePathWidget::applied, [this](){
        p_buttons->button(QDialogButtonBox::Save)->setEnabled(true);
    });
    connect(p_buttons->button(QDialogButtonBox::Cancel), &QAbstractButton::clicked,
            this, &BCSavePathDialog::reject);
    connect(p_buttons->button(QDialogButtonBox::Save), &QAbstractButton::clicked,
            this, &BCSavePathDialog::accept);

    vbl->addWidget(p_buttons);
    setLayout(vbl);
}

void BCSavePathDialog::accept()
{
    p_widget->save();
    return QDialog::accept();
}

void BCSavePathDialog::reject()
{
    SettingsStorage s;
    auto sp = s.get(BC::Key::savePath, QString(""));
    auto en = s.get(BC::Key::exptNum, 0) + 1;
    if(sp.isEmpty())
    {
        int ret = QMessageBox::critical(this, "Blackchirp Error",
            QString("You must select a valid directory in order to run Blackchirp. Would you like to try again?"),
            QMessageBox::No | QMessageBox::Yes, QMessageBox::Yes);
        if(ret == QMessageBox::No)
            QDialog::reject();
    }
    else
    {
        int ret = QMessageBox::warning(this, "Blackchirp Warning",
            QString("The current save path is set to %1 and the current experiment number is %2. These values will not be changed.\n\nIs this correct?").arg(sp).arg(en),
            QMessageBox::No | QMessageBox::Yes, QMessageBox::Yes);
        if(ret == QMessageBox::Yes)
            QDialog::reject();
    }
}

QSize BCSavePathDialog::sizeHint() const
{
    return {400, 300};
}
