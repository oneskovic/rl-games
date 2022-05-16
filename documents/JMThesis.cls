%% SCThesis.cls

\def\fileversion{0.01}
\def\filedate{06/06/12}

\NeedsTeXFormat{LaTeX2e}
\ProvidesPackage{JMThesis}[\filedate\space\fileversion\space JM thesis]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}

\ProcessOptions

\LoadClass{book}

\def\es{} % eat space for nice formatting

%%
%% REDEFINE PAGESTYLE "HEADINGS"
%%
%% uppercase removed, slanted replaced by boldface, and underline added, ES
\if@twoside         % If two-sided printing.
\def\ps@headings{\let\@mkboth\markboth
%%\def\@oddfoot{\hfill\hbox{\today}\hfill}\def\@evenfoot{\hfill\hbox{\today}\hfill}%       No feet.
\def\@evenhead{\underline{\makebox[\textwidth][l]{\es
  \rm \thepage  \vphantom{g}\hfill \bf \leftmark}}}%
\def\@oddhead{\underline{\makebox[\textwidth][l]{\es
  \hbox{}\bf \rightmark \vphantom{g}\hfill \rm\thepage}}}%
\def\chaptermark##1{\markboth {\ifnum \c@secnumdepth >\m@ne
  \@chapapp\ \thechapter. \ \fi ##1}{}}%
\def\sectionmark##1{\markright {\ifnum \c@secnumdepth >\z@
  \thesection. \ \fi ##1}}}
\else               % If one-sided printing.
\def\ps@headings{\let\@mkboth\markboth
\def\@oddfoot{}\def\@evenfoot{}%     No feet.
\def\@oddhead{\hbox {}\sl \rightmark \hfil \rm\thepage}% Heading.
\def\chaptermark##1{\markright {\uppercase{\ifnum \c@secnumdepth >\m@ne
  \@chapapp\ \thechapter. \ \fi ##1}}}}
\fi
\pagestyle{headings}

% CONTENTS ENTRY FOR FULL SECTION TITLES
%
% The following change to a LATEX.TEX part is for: If \section[a]{bb}
% is used in the original setting, then "a" gets the running head as
% well as the entry to the table of contents. Here we want however
% that "a" gets the running head only and the entry to the table of
% contents is still "bb" (JG).
%
% begin of overwrite

\def\@chap#1#2#3#4#5#6[#7]#8{\ifnum #2>\c@chapnum
     \def\@svchap{}\else 
     \refstepcounter{#1}\edef\@svchap{\csname the#1\endcsname\hskip 1em }\fi
     \@tempskipa #5\relax
      \ifdim \@tempskipa>\z@ 
        \begingroup #6\relax
          \@hangfrom{\hskip #3\relax\@svshap}{\interlinepenalty \@M #8\par}
        \endgroup
       \csname #1mark\endcsname{#7}\addcontentsline
         {toc}{#1}{\ifnum #2>\c@chapnum \else
                      \protect\numberline{\csname the#1\endcsname}\fi
                    #8}\else    % #7 changed to #8, JG
        \def\@svchaphd{#6\hskip #3\@svchap #8\csname #1mark\endcsname
                      {#7}\addcontentsline
                           {toc}{#1}{\ifnum #2>\c@chapnum \else
                             \protect\numberline{\csname the#1\endcsname}\fi
                       #8}}\fi   % #7 changed to #8, JG
     \@xchap{#5}}
% end of overwrite

%
% RUNNING HEAD MACROS
%
% "empty" left and right head, i.e. without "CHAPTER", chapter-
% number and section-number, only the text specified by parameter (JG).
%
\def\runheadel#1{\markboth{#1}{}}
\def\runheader#1{\markright{#1}}
%
% chapter (left) and section (right) heads, text specified by
% parameter (JG).
%
\def\runheadc#1{\markboth{
   \ifnum \c@secnumdepth>\m@ne
   \@chapapp\ \thechapter. \ \fi #1}{}}
\def\runheads#1{\markright{
   \ifnum \c@secnumdepth >\z@
   \thesection. \ \fi #1}}

%
% DEFINE AND SET FLAG FOR ADDING CHAPER*-TITLES TO THE CONTENS
%
\newif\ifaddcontents \addcontentsfalse    % Create flag and set false.

%
% REDEFINITIONS TO THE REP10 STYLE
%
% Redefinitions to the rep10.sty can only be executed after \documentstyle.
% The modifications are: \addcontentsline in \def\chapter
% uses #2 instead of #1 to make #1 being still the running
% head, but #2 being written into the table of contents. (JG)
% \doubleclearpage is used in \chapter and \chapter*. Depending on the
% addcontents flag, \addcontentsline is executed for \chapter*.
% Set running heads for \chapter* as well. (ES)
%
