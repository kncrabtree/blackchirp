#include "experimentsummarywidget.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QTreeView>
#include <QHeaderView>

#include <data/experiment/experiment.h>
#include <data/model/exptsummarymodel.h>

ExperimentSummaryWidget::ExperimentSummaryWidget(QWidget *parent) : QWidget(parent)
{
    auto vbl = new QVBoxLayout;

    auto lbl = new QLabel("Filter");
    p_le = new QLineEdit;

    auto hbl = new QHBoxLayout;
    hbl->addWidget(lbl,0);
    hbl->addWidget(p_le,1);
    vbl->addLayout(hbl,0);

    p_view = new QTreeView;
    vbl->addWidget(p_view,1);

    setLayout(vbl);
}

void ExperimentSummaryWidget::setExperiment(Experiment *exp)
{
    if(p_view->model())
        p_view->model()->deleteLater();

    auto proxy = new ExptProxyModel;
    auto model = new ExptSummaryModel(exp,proxy);
    proxy->setSourceModel(model);
    p_view->setModel(proxy);
    p_view->header()->setSectionResizeMode(QHeaderView::Stretch);
    connect(p_le,&QLineEdit::textEdited,
            [proxy](QString s){
        QRegularExpression exp(s,QRegularExpression::CaseInsensitiveOption);
        proxy->setFilterRegularExpression(exp);
    });
    p_view->setSortingEnabled(true);
    p_view->sortByColumn(0,Qt::AscendingOrder);
}


QSize ExperimentSummaryWidget::sizeHint() const
{
    return {500,600};
}
