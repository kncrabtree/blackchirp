#include "experimentviewwidget.h"

#include <QTabWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QTableWidget>
#include <QHeaderView>
#include <QLabel>

ExperimentViewWidget::ExperimentViewWidget(int num, QWidget *parent) : QWidget(parent)
{
    d_experiment = Experiment(num);
    setWindowFlags(Qt::Window);
    setWindowTitle(QString("Experiment %1").arg(num));
    setAttribute(Qt::WA_DeleteOnClose);

    QHBoxLayout *hbl = new QHBoxLayout;

    if(d_experiment.number() < 1)
    {
        QLabel *errLabel = new QLabel(d_experiment.errorString());
        errLabel->setAlignment(Qt::AlignCenter);
        errLabel->setWordWrap(true);
        hbl->addWidget(errLabel);
        resize(500,500);
        setLayout(hbl);
        return;
    }

    p_tabWidget = new QTabWidget(this);
    p_tabWidget->setTabPosition(QTabWidget::East);

    QWidget *hdr = new QWidget();
    QVBoxLayout *hdrvl = new QVBoxLayout();

    QTableWidget *tw = new QTableWidget(this);
    tw->setColumnCount(3);
    tw->setEditTriggers(QTableWidget::NoEditTriggers);
    tw->setSelectionBehavior(QAbstractItemView::SelectRows);

    tw->setHorizontalHeaderItem(0,new QTableWidgetItem(QString("Key")));
    tw->setHorizontalHeaderItem(1,new QTableWidgetItem(QString("Value")));
    tw->setHorizontalHeaderItem(2,new QTableWidgetItem(QString("Unit")));
    tw->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);

    auto header = d_experiment.headerMap();
    auto it = header.constBegin();
    tw->setRowCount(header.size());
    for(int i=0; it != header.constEnd(); i++, it++)
    {
        tw->setItem(i,0,new QTableWidgetItem(it.key()));
        tw->setItem(i,1,new QTableWidgetItem(it.value().first.toString()));
        tw->setItem(i,2,new QTableWidgetItem(it.value().second));
    }
    hdrvl->addWidget(tw);
    hdr->setLayout(hdrvl);

    p_tabWidget->addTab(hdr,QString("Header"));
    tw->resizeColumnsToContents();

    hbl->addWidget(p_tabWidget);
    setLayout(hbl);
}

