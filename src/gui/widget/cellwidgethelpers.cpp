#include "cellwidgethelpers.h"

#include <QHBoxLayout>
#include <QTableWidget>
#include <QWidget>

namespace BC::Gui {

void centerCellWidget(QTableWidget *table, int row, int col, QWidget *w)
{
    auto wrap = new QWidget;
    auto h = new QHBoxLayout(wrap);
    h->setContentsMargins(0,0,0,0);
    h->addWidget(w);
    h->setAlignment(Qt::AlignCenter);
    table->setCellWidget(row,col,wrap);
}

} // namespace BC::Gui
