#include "aboutdialog.h"

#include <QApplication>
#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QPushButton>
#include <QSysInfo>
#include <QTabWidget>
#include <QTableWidget>
#include <QShowEvent>
#include <QUrl>
#include <QVBoxLayout>

#include <gui/style/themecolors.h>

using namespace Qt::Literals::StringLiterals;

namespace {
constexpr auto k_githubUrl  = "https://github.com/kncrabtree/blackchirp";
constexpr auto k_docsUrl    = "https://blackchirp.readthedocs.io/";
constexpr auto k_discordUrl = "https://discord.gg/88CkbAKUZY";
}

AboutDialog::AboutDialog(const AppInfo &info, QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle(u"About "_s + info.name);
    setWindowIcon(ThemeColors::createThemedIcon(
        QString(":/icons/bc_logo_trans.svg"), ThemeColors::IconPrimary, this));
    setMinimumWidth(480);

    auto *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(12);

    // Header: icon + name, version, build
    {
        auto *row = new QHBoxLayout;

        auto *iconLabel = new QLabel(this);
        iconLabel->setPixmap(
            ThemeColors::createThemedIcon(
                QString(":/icons/bc_logo_trans.svg"), ThemeColors::IconPrimary, this)
            .pixmap(64, 64));
        row->addWidget(iconLabel);

        auto *col = new QVBoxLayout;
        auto *nameLabel = new QLabel(info.name, this);
        {
            auto f = nameLabel->font();
            f.setPointSize(f.pointSize() + 4);
            f.setBold(true);
            nameLabel->setFont(f);
        }
        col->addWidget(nameLabel);
        col->addWidget(new QLabel(u"Version "_s + info.version, this));
        col->addWidget(new QLabel(u"Build: "_s + info.build, this));

        row->addLayout(col);
        row->addStretch();
        mainLayout->addLayout(row);
    }

    auto *tabs = new QTabWidget(this);

    // Overview tab
    {
        auto *w = new QWidget;
        auto *vl = new QVBoxLayout(w);
        vl->setSpacing(8);

        auto *desc = new QLabel(info.description, w);
        desc->setWordWrap(true);
        vl->addWidget(desc);

        auto *copy = new QLabel(
            u"Copyright © Kyle N. Crabtree <kncrabtree@ucdavis.edu>, University of California, Davis"_s, w);
        copy->setWordWrap(true);
        vl->addWidget(copy);

        auto *lic = new QLabel(
            u"Licensed under the "
             "<a href=\"https://opensource.org/licenses/MIT\">MIT License</a>."_s,
            w);
        lic->setOpenExternalLinks(true);
        lic->setWordWrap(true);
        vl->addWidget(lic);

        vl->addStretch();

        auto *linksBox = new QGroupBox("Online Resources"_L1, w);
        auto *hl = new QHBoxLayout(linksBox);

        auto addLink = [&](QLatin1StringView label, const char *url) {
            auto *btn = new QPushButton(label, linksBox);
            connect(btn, &QPushButton::clicked, linksBox, [url]() {
                QDesktopServices::openUrl(QUrl(QLatin1StringView(url)));
            });
            hl->addWidget(btn);
        };
        addLink("GitHub"_L1, k_githubUrl);
        addLink("Documentation"_L1, k_docsUrl);
        addLink("Discord"_L1, k_discordUrl);

        vl->addWidget(linksBox);
        tabs->addTab(w, "Overview"_L1);
    }

    // Third-Party Libraries tab
    {
        auto *w = new QWidget;
        auto *vl = new QVBoxLayout(w);

        struct LibRow { QString name, version, license; };
        const QList<LibRow> libs = {
            {"Qt"_L1,                     QLatin1StringView(qVersion()),         "LGPL 3.0"_L1},
            {"Qwt"_L1,                    QLatin1StringView(BC_QWT_VERSION),     "Modified LGPL 2.1"_L1},
            {"GNU Scientific Library"_L1, QLatin1StringView(BC_GSL_VERSION),     "GPL 3.0"_L1},
            {"Eigen3"_L1,                 QLatin1StringView(BC_EIGEN3_VERSION),  "MPL 2.0"_L1},
        };

        auto *table = new QTableWidget(static_cast<int>(libs.size()), 3, w);
        table->setHorizontalHeaderLabels({"Library"_L1, "Version"_L1, "License"_L1});
        table->horizontalHeader()->setStretchLastSection(true);
        table->verticalHeader()->setVisible(false);
        table->setEditTriggers(QAbstractItemView::NoEditTriggers);
        table->setSelectionMode(QAbstractItemView::NoSelection);
        table->setAlternatingRowColors(true);
        table->setFocusPolicy(Qt::NoFocus);

        for(int i = 0; i < libs.size(); ++i) {
            table->setItem(i, 0, new QTableWidgetItem(libs[i].name));
            table->setItem(i, 1, new QTableWidgetItem(libs[i].version));
            table->setItem(i, 2, new QTableWidgetItem(libs[i].license));
        }
        table->resizeColumnsToContents();
        vl->addWidget(table);
        tabs->addTab(w, "Third-Party Libraries"_L1);
    }

    // Build Info tab
    {
        auto *w = new QWidget;
        auto *fl = new QFormLayout(w);
        fl->setLabelAlignment(Qt::AlignRight | Qt::AlignVCenter);

        auto addRow = [&](const QString &label, const QString &value) {
            auto *lbl = new QLabel(value, w);
            lbl->setTextInteractionFlags(Qt::TextSelectableByMouse);
            fl->addRow(label, lbl);
        };

        addRow("Qt version:"_L1,       QLatin1StringView(QT_VERSION_STR));
        addRow("Platform:"_L1,         QSysInfo::prettyProductName());
        addRow("CPU architecture:"_L1, QSysInfo::currentCpuArchitecture());

        for(const auto &[key, val] : info.features)
            addRow(key + QLatin1Char(':'), val);

        tabs->addTab(w, "Build Info"_L1);
    }

    mainLayout->addWidget(tabs);

    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Close, this);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    mainLayout->addWidget(buttonBox);
}

void AboutDialog::showEvent(QShowEvent *event)
{
    QDialog::showEvent(event);
    setMinimumSize(size());
}
